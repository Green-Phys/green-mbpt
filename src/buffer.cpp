#include<iostream>
#include"green/mbpt/buffered_reader/buffer.hpp"
#include <unistd.h>


int buffer::n_buffer_elem_heuristics(double ratio, int element_size, int total_num_elem) {
    //figure out how much memory is available on the machine
    unsigned long long pages = sysconf(_SC_PHYS_PAGES);
    unsigned long long page_size = sysconf(_SC_PAGE_SIZE);
    unsigned long long total_memory=pages*page_size;
    //if(verbose_) std::cout<<"total memory size: "<<total_memory/(1024.*1024.*1024.)<<" GB"<<std::endl;

    //figure out how many elements we could fit total
    unsigned long long total_elements=total_memory/element_size;

    //modify by proportion of memory, round and return
    int proposed_nelem=(int)(total_elements*ratio);


    if(proposed_nelem >= total_num_elem) proposed_nelem=total_num_elem;
    //if(verbose_ && (proposed_nelem <100)) std::cerr<<"WARNING: ONLY "<<proposed_nelem<<" (<100) buffer elements fit to memory."<<std::endl; 

    return proposed_nelem;
}
void buffer::setup_mpi_shmem(){
  //split communicator for shared memory MPI
  int global_rank; MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
  MPI_Info info; MPI_Info_create(&info);
  MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,global_rank,info,&shmem_comm_);
  MPI_Comm_size(shmem_comm_,&shmem_size_);
  MPI_Comm_rank(shmem_comm_,&shmem_rank_);

  //print shared memory MPI info
  MPI_Barrier(MPI_COMM_WORLD);
  int global_size; MPI_Comm_size(MPI_COMM_WORLD, &global_size);
  for(int i=0; i<global_size;++i){
    MPI_Barrier(MPI_COMM_WORLD);
    if(i==global_rank && verbose_)
      std::cout<<"global rank "<<i<<" is local shmem rank: "<<shmem_rank_<<" with shmem size: "<<shmem_size_<<std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  //create a shared memory status for the elements
  element_status_.setup_shmem_region(shmem_comm_, number_of_keys_, true); //also validate shmem model
  //initialize status on shmem rank 0
  if(shmem_rank_==0) for(int i=0;i<number_of_keys_;++i) element_status_[i]=status_elem_unavailable;

  //create a shared memory array counting how many threads use this buffer
  buffer_access_counter_.setup_shmem_region(shmem_comm_, number_of_buffered_elements_);
  //initialize status on shmem rank 0
  if(shmem_rank_==0) for(int i=0;i<number_of_buffered_elements_;++i) buffer_access_counter_[i]=0;

  //create a shared memory array with an index of when this buffer was last requested
  //buffer_last_access_.setup_shmem_region(shmem_comm_, number_of_buffered_elements_);
  //initialize on shmem rank 0
  //if(shmem_rank_==0) for(int i=0;i<number_of_buffered_elements_;++i) buffer_last_access_[i]=buffer_never_accessed;

  //create a shared memory access array pointing from buffer to element
  buffer_key_.setup_shmem_region(shmem_comm_, number_of_buffered_elements_);
  //initialize on shmem rank 0
  if(shmem_rank_==0) for(int i=0;i<number_of_buffered_elements_;++i) buffer_key_[i]=key_index_nowhere;

  //create a shared memory array pointing from element to buffer
  element_buffer_index_.setup_shmem_region(shmem_comm_, number_of_keys_);
  //initialize on shmem rank 0
  if(shmem_rank_==0) for(int i=0;i<number_of_keys_;++i) element_buffer_index_[i]=buffer_index_nowhere;

  //create shared memory window that we'll use for a simple lock in case we only allow a single thread to read at a time
  single_thread_readlock_.setup_shmem_region(shmem_comm_, 1);

  //finally the allocation of the buffer
  buffer_data_.resize(number_of_buffered_elements_);
  for(int i=0;i<number_of_buffered_elements_;++i)
    buffer_data_[i].setup_shmem_region(shmem_comm_,(unsigned long long) element_size_);

  MPI_Barrier(shmem_comm_);
}
void buffer::release_mpi_shmem(){
  //nothing to be done since we're releasing inside of the shared memory region now. 
}
void buffer::release_element(int key){
  //lock access counter
  element_status_.acquire_exclusive_lock();
  buffer_access_counter_.acquire_exclusive_lock();

  //consistency checks
  if(element_status_[key]!=status_elem_available) throw std::logic_error("element released during processing.");
  if(buffer_access_counter_[element_buffer_index_[key]]<=0) throw std::logic_error("buffer released more than acquired.");
  //decrease access counter
  buffer_access_counter_[element_buffer_index_[key]]--;

  //unlock access counter
  element_status_.release_exclusive_lock();
  buffer_access_counter_.release_exclusive_lock();
}
const double *buffer::access_element(int key){
  ctr_total_access_++;

  //lock status
  element_status_.acquire_exclusive_lock();

  //check if another thread is reading this element. If so just wait until the data has arrived.
  if(element_status_[key]==status_elem_reading){
    ctr_read_delay_++;
    while(element_status_[key]==status_elem_reading){
      element_status_.release_exclusive_lock();
      //should sleep for 1 millisecond to wait for data to arrive
      std::this_thread::sleep_for(std::chrono::milliseconds(1)); //go to sleep for one millisecond, then check again
      element_status_.acquire_exclusive_lock();
    }
  }
  if(element_status_[key]==status_elem_available){
    //find corresponding buffer
    int buffer=element_buffer_index_[key];
    //lock the access counter and increase by one, then free lock
    buffer_access_counter_.acquire_exclusive_lock();
    buffer_access_counter_[buffer]++;
    buffer_access_counter_.release_exclusive_lock();

    //mark that buffer as recently accessed
    aob_.promote_to_top(buffer);

    //release status lock, return buffer index
    element_status_.release_exclusive_lock();
    return &(buffer_data_[buffer][0]);
  }

  //otherwise status is unavailable and we need to read from file system.
  ctr_fs_read_++; 
  if(element_status_[key]!=status_elem_unavailable) throw std::runtime_error("unknown element status");

  //set status to reading
  element_status_[key]=status_elem_reading;

  //find key of oldest unused buffer 
  int buffer=aob_.oldest_entry();
  int old_key=buffer_key_[buffer];
  {
    //set data for old key to unavailable
    if(old_key!=key_index_nowhere){
      ctr_buffer_eviction_++;
      element_status_[old_key]=status_elem_unavailable;
    }
    //sanity check
    if(buffer_access_counter_[buffer]!=0) throw std::logic_error("freeing buffer still in use");
    
    //repurpose buffer for current key
    element_buffer_index_[key]=buffer;

    buffer_access_counter_.acquire_exclusive_lock();
    buffer_access_counter_[buffer]++;
    buffer_access_counter_.release_exclusive_lock();

    //age out the oldest buffer and move it to the top
    aob_.replace_oldest_entry(buffer);

    buffer_key_.acquire_exclusive_lock();
    buffer_key_[buffer]=key;
    buffer_key_.release_exclusive_lock();
  }

  //unlock status. Nobody will mess with our buffer cause we are currently in status 'reading'
  element_status_.release_exclusive_lock();

  //go and read the data
  double *read_buffer= &(buffer_data_[buffer][0]);
  if(single_thread_read_)
    single_thread_readlock_.acquire_exclusive_lock();
  //std::cout<<"rank: "<<shmem_rank()<<" reading key: "<<key<<" into buffer: "<<buffer<<std::endl;
  reader_ptr_->read_key(key, read_buffer);
  if(single_thread_read_)
    single_thread_readlock_.release_exclusive_lock();

  // lock, change status to available, and release lock 
  element_status_.acquire_exclusive_lock();
  element_status_[key]=status_elem_available;
  element_status_.release_exclusive_lock();

  return  read_buffer;
}
