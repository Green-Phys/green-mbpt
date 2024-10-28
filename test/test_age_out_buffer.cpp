#include "gtest/gtest.h"
#include "age_out_buffer.hpp"
#include <mpi.h>
#include <chrono>
#include <thread>

TEST(age_out_buffer, Init) {
  int N=10;
  age_out_buffer aob(N);
}
TEST(age_out_buffer, last) {
  age_out_buffer aob(10);

  MPI_Barrier(MPI_COMM_WORLD);
  EXPECT_EQ(aob.oldest_entry(), 9);
  MPI_Barrier(MPI_COMM_WORLD);
  aob.promote_to_top(8);
  MPI_Barrier(MPI_COMM_WORLD);
  EXPECT_EQ(aob.oldest_entry(), 9);
  MPI_Barrier(MPI_COMM_WORLD);
  aob.promote_to_top(9);
  MPI_Barrier(MPI_COMM_WORLD);
  EXPECT_EQ(aob.oldest_entry(), 7);
  MPI_Barrier(MPI_COMM_WORLD);
  aob.promote_to_top(6);
  MPI_Barrier(MPI_COMM_WORLD);
  EXPECT_EQ(aob.oldest_entry(), 7);
  MPI_Barrier(MPI_COMM_WORLD);
  aob.promote_to_top(7);
  MPI_Barrier(MPI_COMM_WORLD);
  EXPECT_EQ(aob.oldest_entry(), 5);
  MPI_Barrier(MPI_COMM_WORLD);
  aob.promote_to_top(9);
  MPI_Barrier(MPI_COMM_WORLD);
  EXPECT_EQ(aob.oldest_entry(), 5);
  MPI_Barrier(MPI_COMM_WORLD);
  
}
TEST(age_out_buffer, repeat) {
  age_out_buffer aob(10);

  aob.promote_to_top(8);
  aob.promote_to_top(8);
  aob.promote_to_top(8);
  aob.promote_to_top(8);

  EXPECT_EQ(aob.oldest_entry(), 9);

}
