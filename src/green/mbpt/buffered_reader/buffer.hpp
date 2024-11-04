#pragma once
#include<Eigen/Dense>
#include<mpi.h>
#include"shared_memory_region.hpp"
#include"access_counter.hpp"
#include"reader.hpp"
#include"age_out_buffer.hpp"

enum element_status{
  status_elem_reading=-2,
  status_elem_unavailable=-1,
  status_elem_available=0
};
enum{
  buffer_never_accessed=-1
};
enum{
  buffer_index_nowhere=-1,
  key_index_nowhere=-1
};

class buffer{
public:
  buffer(int element_size_in_doubles, int number_of_keys, int number_of_buffered_elements, reader *reader_ptr, bool verbose=false, bool single_thread_read=false):
    element_size_(element_size_in_doubles),
    number_of_keys_(number_of_keys),
    number_of_buffered_elements_(number_of_buffered_elements),
    verbose_(verbose),
    aob_(number_of_buffered_elements_),
    reader_ptr_(reader_ptr),
    single_thread_read_(single_thread_read)
  { 
    setup_mpi_shmem();
  }
  ~buffer(){
    MPI_Barrier(shmem_comm_);
    for(int i=0;i<number_of_buffered_elements_;++i){
      if (buffer_access_counter_[i]!=0){
        std::cerr<<"buffer released is still being accessed"<<std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }
    MPI_Barrier(shmem_comm_);
    release_mpi_shmem();
  }
  //getter function for size of each buffer element
  std::size_t element_size() const{return element_size_;}
  //getter function for total number of keys
  std::size_t number_of_keys() const{return number_of_keys_;}

  //memory heuristics for figuring out how many elements we should allocate, as function of total memory size
  static int n_buffer_elem_heuristics(double ratio, int element_size_in_bytes, int total_num_elem) ;

  int element_status(int key) const{ return element_status_[key];}

  //as a user: this is how you access an element. reading will be done behind the scenes
  const double *access_element(int key);
  //as a user: notify that you're done with reading so memory can be reused, if idle.
  void release_element(int key);

  //shmem rank and size
  int shmem_rank() const{ return shmem_rank_;}
  int shmem_size() const{ return shmem_size_;}
private:
  void setup_mpi_shmem();
  void release_mpi_shmem();

   //returns the key and buffer of the oldest buffer that is unused.
//  std::pair<int, int> find_oldest_unused_buffer_key() const;

  //amount of memory each element uses (in units of doubles)
  const std::size_t element_size_;
  //amount of total elements we have
  const std::size_t number_of_keys_;
  //amount of elements we can buffer
  const std::size_t number_of_buffered_elements_;

  const bool verbose_;

  //this is where we do the accounting and locking/unlocking.
  shared_memory_region<int> element_status_;
  //this is where we count how many concurrent accesses we have
  shared_memory_region<int> buffer_access_counter_;
  //this is where we check when this element was last accessed
  //shared_memory_region<unsigned long long> buffer_last_access_;
  //this is where we store the key for a particular buffer
  shared_memory_region<unsigned long long> buffer_key_;
  //this is where we store the buffer for a particular key
  shared_memory_region<int> element_buffer_index_;
  //this is where we keep the actual data
  std::vector<shared_memory_region<double>> buffer_data_;

  //in case we only allow a single thread to read at once
  shared_memory_region<bool> single_thread_readlock_;
  bool single_thread_read_;

  access_counter ctr_;
  age_out_buffer aob_;

  //MPI shared memory auxiliaries
  MPI_Comm shmem_comm_;
  int shmem_size_, shmem_rank_;


  //pointer to the object that does the actual reading
  reader *reader_ptr_;
};
