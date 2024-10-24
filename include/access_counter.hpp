#pragma once
#include<Eigen/Dense>
#include<mpi.h>
#include"shared_memory_region.hpp"

class access_counter{
public:
  access_counter(){
    //split communicator for shared memory MPI
    int global_rank; MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    MPI_Info info; MPI_Info_create(&info);
    MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,global_rank,info,&shmem_comm_);
    int shmem_rank; MPI_Comm_rank(shmem_comm_, &shmem_rank);
 
    //create a shared memory for the ctr
    ctr_.setup_shmem_region(shmem_comm_, 1);
    if(shmem_rank==0) ctr_[0]=0; //initialize
    MPI_Barrier(shmem_comm_); //sync
  }
  unsigned long long operator()()const{ return ctr_[0];}
  access_counter & operator++(int){
    ctr_.acquire_exclusive_lock();
    ctr_[0]++;
    ctr_.release_exclusive_lock();
    return *this;
  } 
  int shmem_size() const{
    int size; MPI_Comm_size(shmem_comm_, &size);
    return size;
  }
private:
  shared_memory_region<unsigned long long> ctr_;
  MPI_Comm shmem_comm_;
};
