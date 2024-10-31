#include <catch2/catch_test_macros.hpp>
#include "access_counter.hpp"
#include <mpi.h>
#include <chrono>
#include <thread>

TEST_CASE("Init","[access_counter]") {
  access_counter ctr;
  REQUIRE(ctr()==0);
}
TEST_CASE("count", "[access_counter]") {
  access_counter ctr;
  int ncount=100;

  for(int i=0;i<ncount;++i){
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ctr++;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  REQUIRE(ctr()==ctr.shmem_size()*ncount);
}
