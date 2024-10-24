#pragma once
#include<Eigen/Dense>
#include<mpi.h>

template<typename T> class shared_memory_region{
public:
  shared_memory_region(){allocated_=false;locked_=false;}
  void setup_shmem_region(const MPI_Comm &shmem_comm, std::size_t region_size){
    if(allocated_) throw std::logic_error("shmem region can only be allocated once");
    int shmem_size, shmem_rank;
    MPI_Comm_size(shmem_comm,&shmem_size);
    MPI_Comm_rank(shmem_comm,&shmem_rank);
    //create a shared memory status buffer
    MPI_Win_allocate_shared(shmem_rank==0?region_size* sizeof(T):0, sizeof(T), MPI_INFO_NULL, shmem_comm, &buffer_status_alloc_, &window_);
    //get a local pointer to shared memory buffer
    {
      MPI_Aint rss2;
      int soT2;
      MPI_Win_shared_query(window_, 0, &rss2, &soT2, &buffer_status_);
      if(rss2!=region_size*sizeof(T)) throw std::runtime_error("shared window error: nk2 should be number of keys");
      if(soT2!=sizeof(T)) throw std::runtime_error("shared window error: soi2 should be sizeof(int)");
    }
    region_size_=region_size;
    allocated_=true;
  }
  ~shared_memory_region(){
    if(allocated_)
      MPI_Win_free(&window_);
    if(locked_){
      std::cerr<<"Memory region locked in destructor"<<std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); //take down everybody
    }
  }
  //const access to shmem vector
  const T& operator[](size_t idx)const{
    return *(buffer_status_+idx);
  }
  //non-const access to shmem vector
  T& operator[](size_t idx){
    return *(buffer_status_+idx);
  }
  const std::size_t &size() const{ return region_size_;}
  bool allocated() const { return allocated_;}

  void acquire_exclusive_lock(){
    if(locked_) throw std::runtime_error("lock already acquired");
      MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0,0, window_);
    locked_=true;
  }
  void release_exclusive_lock(){
    if(!locked_) throw std::runtime_error("trying to release lock that has not been acquired");
      MPI_Win_unlock(0, window_);
    locked_=false;
  }
  //lock getter function
  bool locked() const{ return locked_;}

private:
  MPI_Win window_; //the MPI window where we keep read/write/availability accounting info
  T *buffer_status_; //pointer to be addressed with shared mem MPI
  T *buffer_status_alloc_; //pointer to be allocated and deallocated with shared mem MPI
  std::size_t region_size_; //size of shared memory region
  bool allocated_; //to be set after memory obtained from system
  bool locked_;
};

