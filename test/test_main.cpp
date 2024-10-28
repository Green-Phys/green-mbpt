#include <iostream>
#include "gtest/gtest.h"
#include<mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int size=0;
  int rank=0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if(!rank) std::cout<<"running tests under MPI on "<<size<<" processes. Only printing results from rank 0."<<std::endl;


  ::testing::InitGoogleTest(&argc, argv);
  ::testing::TestEventListeners& listeners =::testing::UnitTest::GetInstance()->listeners();
  if (rank != 0) {
    delete listeners.Release(listeners.default_result_printer());
  }

  int rat=RUN_ALL_TESTS();

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return rat;
}


