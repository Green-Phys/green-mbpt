#pragma once
#include<Eigen/Dense>
#include<mpi.h>

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
    verbose_(verbose){    
    setup_mpi_shmem();
    buffer_status_=Eigen::VectorXi::Constant(number_of_keys, status_elem_unavailable);
  }
  //getter function for size of each buffer element
  std::size_t element_size() const{return element_size_;}
  //getter function for total number of keys
  std::size_t number_of_keys() const{return number_of_keys_;}

  //memory heuristics for figuring out how many elements we should allocate, as function of total memory size
  int nelem_heuristics(double ratio) const;

  int buffer_status(int key) const{ return buffer_status_[key];}
private:
  void setup_mpi_shmem();

  const std::size_t element_size_;
  const std::size_t number_of_keys_;

  const bool verbose_;

  Eigen::VectorXi buffer_status_;


  //MPI shared memory auxiliaries
  MPI_Comm shmem_comm_;
  int shmem_size_, shmem_rank_;

};
