#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "shared_memory_region.hpp"
#include <mpi.h>
#include <chrono>
#include <thread>

TEST_CASE("Init","[shmem]") {
  shared_memory_region<int> shmemi;
  shared_memory_region<double> shmemd;
  REQUIRE_FALSE(shmemi.allocated());
  REQUIRE_FALSE(shmemd.allocated());
}
TEST_CASE("Alloc","[shmem]") {
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
    REQUIRE(shmemi[i]==i);
    REQUIRE_THAT(shmemd[i], Catch::Matchers::WithinAbs(i*M_PI, 1.e-14));
  } 
  //make sure we're allocated
  REQUIRE(shmemi.allocated());
  REQUIRE(shmemd.allocated());

  //double allocation throw
  REQUIRE_THROWS_AS(shmemi.setup_shmem_region(shmem_comm, 1), std::logic_error);
  REQUIRE_THROWS_AS(shmemd.setup_shmem_region(shmem_comm, 1), std::logic_error);
}
TEST_CASE("Lock","[shmem]") {
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
  REQUIRE(duration.count()>shmem_size*1000);
}
