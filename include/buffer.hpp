#pragma once
#include<Eigen/Dense>
#include<mpi.h>
#include"shared_memory_region.hpp"

enum buffer_status{
  status_elem_reading=-2,
  status_elem_unavailable=-1,
  status_elem_idle=0
};

class buffer{
public:
  buffer(int element_size, int number_of_keys, bool verbose=false):
    element_size_(element_size),
    number_of_keys_(number_of_keys),
    verbose_(verbose)
  { 
    setup_mpi_shmem();
  }
  ~buffer(){
    release_mpi_shmem();
  }
  //getter function for size of each buffer element
  std::size_t element_size() const{return element_size_;}
  //getter function for total number of keys
  std::size_t number_of_keys() const{return number_of_keys_;}

  //memory heuristics for figuring out how many elements we should allocate, as function of total memory size
  static int nelem_heuristics(double ratio, int element_size, int total_num_elem) ;

  int buffer_status(int key) const{ return buffer_status_[key];}
private:
  void setup_mpi_shmem();
  void release_mpi_shmem();

  const std::size_t element_size_;
  const std::size_t number_of_keys_;

  const bool verbose_;

  //this is where we do the accounting and locking/unlocking.
  shared_memory_region<int> buffer_status_;
  //this is where we keep the actual data
  shared_memory_region<double> buffer_data_;


  //MPI shared memory auxiliaries
  MPI_Comm shmem_comm_;
  int shmem_size_, shmem_rank_;

};
