#define CATCH_CONFIG_RUNNER
#include <iostream>
#include <catch2/catch_session.hpp>
#include<mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int size=0;
  int rank=0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if(!rank) std::cout<<"running tests under MPI on "<<size<<" processes. Only printing results from rank 0."<<std::endl;


  int result = Catch::Session().run( argc, argv );

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return result;
}


