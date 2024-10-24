#include "gtest/gtest.h"
#include "access_counter.hpp"
#include <mpi.h>
#include <chrono>
#include <thread>

TEST(access_counter, Init) {
  access_counter ctr;
  EXPECT_EQ(ctr(), 0);
}
TEST(access_counter, count) {
  access_counter ctr;
  int ncount=100;

  for(int i=0;i<ncount;++i){
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ctr++;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  EXPECT_EQ(ctr(), ctr.shmem_size()*ncount);
}
