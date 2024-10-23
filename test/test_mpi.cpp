#include "gtest/gtest.h"
#include <mpi.h>

TEST(MPI, Size) {
  int size=0;
  int rank=0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  EXPECT_GT(size,2);  //if this throws run the tests with mpirun -np X, X>=2. 
}
