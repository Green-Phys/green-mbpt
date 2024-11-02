#pragma once
#include<iostream>
#include<mpi.h>
#include<vector>
#include"shared_memory_region.hpp"


class age_out_buffer{
public:
  age_out_buffer(int N):
  N_(N),
  tmp_(N){
    //split communicator for shared memory MPI
    int global_rank; MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    MPI_Info info; MPI_Info_create(&info);
    MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,global_rank,info,&shmem_comm_);
    int shmem_rank; MPI_Comm_rank(shmem_comm_, &shmem_rank);

    //create a shared memory for the ctr
    buffer_.setup_shmem_region(shmem_comm_, N_);
    if(shmem_rank==0){
      for(int i=0;i<N;++i)
        buffer_[i]=i;
    } //initialize
    MPI_Barrier(shmem_comm_); //sync

  }
  ~age_out_buffer(){
    buffer_.acquire_exclusive_lock();
    validate();
    buffer_.release_exclusive_lock();
    MPI_Barrier(shmem_comm_);
  }
  //
  void promote_to_top(int key){
    buffer_.acquire_exclusive_lock();
    //find the key
    bool found=false;
    if(buffer_[0]!=key){
      tmp_[0]=key;
      for(int i=1;i<N_;++i){
        tmp_[i]=found?buffer_[i]:buffer_[i-1];
        if(buffer_[i]==key) found=true;
      }
      if(!found){ 
        std::cerr<<"age_out_buffer: key not found."<<std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      std::copy(tmp_.begin(), tmp_.end(), &buffer_[0]);
    }
    buffer_.release_exclusive_lock();
  }
  int oldest_entry(){
    buffer_.acquire_exclusive_lock();
    int b=buffer_[N_-1];
    buffer_.release_exclusive_lock();
    return b;
  }
  void replace_oldest_entry(int key){
    buffer_.acquire_exclusive_lock();
    for(int i=N_-1;i>0;--i){
      buffer_[i]=buffer_[i-1];
    }
    buffer_[0]=key;
    buffer_.release_exclusive_lock();
  }
private:
  void validate(){
    std::vector<int> b(N_);
    std::copy(&buffer_[0], &buffer_[0]+N_, b.begin());
    std::sort(b.begin(), b.end());
    /*for(int i=0;i<N_;++i){
      std::cout<<i<<" "<<buffer_[i]<<std::endl;
    }
    std::cout<<std::endl;*/
    for(int i=0;i<N_;++i){
      if(b[i]!=i) throw std::runtime_error("buffer invalid.");
    }
  }
  int N_;

  shared_memory_region<int> buffer_;
  std::vector<int> tmp_;
  MPI_Comm shmem_comm_;

};
