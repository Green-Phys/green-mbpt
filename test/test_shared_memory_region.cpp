#include "gtest/gtest.h"
#include "shared_memory_region.hpp"
#include <mpi.h>
#include <chrono>
#include <thread>

TEST(shmem, Init) {
  shared_memory_region<int> shmemi;
  shared_memory_region<double> shmemd;
  EXPECT_FALSE(shmemi.allocated());
  EXPECT_FALSE(shmemd.allocated());
}
TEST(shmem, Alloc) {
  int global_rank; MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
  int shmem_size, shmem_rank;
  MPI_Info info; MPI_Info_create(&info);
  MPI_Comm shmem_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,global_rank,info,&shmem_comm);
  MPI_Comm_size(shmem_comm,&shmem_size);
  MPI_Comm_rank(shmem_comm,&shmem_rank);

  shared_memory_region<int> shmemi;
  shared_memory_region<double> shmemd;
  shmemi.setup_shmem_region(shmem_comm, shmem_size);
  shmemd.setup_shmem_region(shmem_comm, shmem_size);

  //async write
  MPI_Barrier(shmem_comm);
  shmemi[shmem_rank]=shmem_rank;
  shmemd[shmem_rank]=shmem_rank*M_PI;
  MPI_Barrier(shmem_comm);

  //shmem read
  for(int i=0;i<shmem_size;++i){
    EXPECT_EQ(shmemi[i], i);
    EXPECT_NEAR(shmemd[i], i*M_PI, 1.e-14);
  } 
  //make sure we're allocated
  EXPECT_TRUE(shmemi.allocated());
  EXPECT_TRUE(shmemd.allocated());

  //double allocation throw
  EXPECT_THROW(shmemi.setup_shmem_region(shmem_comm, 1), std::logic_error);
  EXPECT_THROW(shmemd.setup_shmem_region(shmem_comm, 1), std::logic_error);
}
TEST(shmem, Lock) {
  int global_rank; MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
  int shmem_size, shmem_rank;
  MPI_Info info; MPI_Info_create(&info);
  MPI_Comm shmem_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,global_rank,info,&shmem_comm);
  MPI_Comm_rank(shmem_comm,&shmem_rank);
  MPI_Comm_size(shmem_comm,&shmem_size);


  shared_memory_region<int> shmemi;
  shmemi.setup_shmem_region(shmem_comm, 200);

  auto start = std::chrono::high_resolution_clock::now();
  shmemi.acquire_exclusive_lock();
  std::this_thread::sleep_for(std::chrono::milliseconds(1)); 
  shmemi.release_exclusive_lock();
  MPI_Barrier(shmem_comm);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  //if the lock works we'll spend 1000 microseconds in the lock for each rank. If it does not work this should trigger even for 2 MPI ranks
  EXPECT_GT(duration.count(), shmem_size*1000);
}
