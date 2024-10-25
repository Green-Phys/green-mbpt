#include<iostream>
#include"buffer.hpp"
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
  element_status_.setup_shmem_region(shmem_comm_, number_of_keys_);
  //initialize status on shmem rank 0
  if(shmem_rank_==0) for(int i=0;i<number_of_keys_;++i) element_status_[i]=status_elem_unavailable;

  //create a shared memory array counting how many threads use this buffer
  buffer_access_counter_.setup_shmem_region(shmem_comm_, number_of_buffered_elements_);
  //initialize status on shmem rank 0
  if(shmem_rank_==0) for(int i=0;i<number_of_buffered_elements_;++i) buffer_access_counter_[i]=0;

  //create a shared memory array with an index of when this buffer was last requested
  buffer_last_access_.setup_shmem_region(shmem_comm_, number_of_buffered_elements_);
  //initialize on shmem rank 0
  if(shmem_rank_==0) for(int i=0;i<number_of_buffered_elements_;++i) buffer_last_access_[i]=buffer_never_accessed;

  //create a shared memory access array pointing from buffer to element
  buffer_key_.setup_shmem_region(shmem_comm_, number_of_buffered_elements_);
  //initialize on shmem rank 0
  if(shmem_rank_==0) for(int i=0;i<number_of_buffered_elements_;++i) buffer_key_[i]=key_index_nowhere;

  //create a shared memory array pointing from element to buffer
  element_buffer_index_.setup_shmem_region(shmem_comm_, number_of_keys_);
  //initialize on shmem rank 0
  if(shmem_rank_==0) for(int i=0;i<number_of_keys_;++i) element_buffer_index_[i]=buffer_index_nowhere;

  MPI_Barrier(shmem_comm_);
}
void buffer::release_mpi_shmem(){
  //nothing to be done since we're releasing inside of the shared memory region now. 
}
void buffer::release_element(int key){
  //lock access counter

  //decrease access counter

  //unlock access counter
}
const double *buffer::access_element(int key){
  //lock status
  
  //check if we have the element available

  //lock the access counter and increase by one, then free lock

  //lock the last access log and set it to current access #, then free lock

  //get status

  //while status is reading: release status lock, sleep for a millisecond, acquire status lock

  //if status is available: release status lock, return buffer index

  //if status is unavailable: set status to reading

  //find key of oldest unused buffer 

    //set status of that key to unavailable
   
    //set last access of that key to never

    //set buffer index of that key to nowhere
 
  //unlock status

  // ACTUALLY READ DATA to buffer

  // lock status

  // set status to available

  // unlock status

  //return buffer
  return NULL;
}
struct last_access_sorter{
  inline bool operator()(const std::pair<std::pair<int,int>,unsigned long long>&A, const std::pair<std::pair<int,int>,unsigned long long>&B) const{ return A.second - B.second; }
};
std::pair<int, int> buffer::find_oldest_unused_buffer_key() const{
  //form a list of all the keys and when we last accessed them
  std::vector<std::pair<std::pair<int,int>,unsigned long long> > key_and_last_access(number_of_buffered_elements_);
  for(int i=0;i<number_of_buffered_elements_;++i){
    key_and_last_access[i].first.first=buffer_key_[i];
    key_and_last_access[i].first.second=i;
    key_and_last_access[i].second=buffer_last_access_[i];
  }
  //sort them by the last access
  std::sort(key_and_last_access.begin(), key_and_last_access.end(), last_access_sorter());

  //return the first element that is not currently in use
  int key, buffer;
  for(int i=0;i<number_of_buffered_elements_;++i){
    key=key_and_last_access[i].first.first;
    buffer=key_and_last_access[i].first.second;
    int current_access=buffer_access_counter_[buffer];
    if(current_access==0)
      break; //otherwise continue because despite being old, this buffer is busy
  }
  if(buffer_access_counter_[buffer]!=0) throw std::runtime_error("all buffers are currently in use. Should have many more buffers than threads!");
  //consistency checks. should never fail
  if(buffer_key_[buffer]!=key) throw std::logic_error("buffer_key points to wrong key");
  if(element_buffer_index_[key]!=buffer) throw std::logic_error("element_index points to wrong buffer");

  return std::make_pair(key,buffer);
}
