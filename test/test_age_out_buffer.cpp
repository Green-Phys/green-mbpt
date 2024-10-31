#include <catch2/catch_test_macros.hpp>

#include "age_out_buffer.hpp"
#include <mpi.h>
#include <chrono>
#include <thread>

TEST_CASE("Init","[age_out_buffer]") {

  int N=10;
  age_out_buffer aob(N);
}
TEST_CASE("Last","[age_out_buffer]") {
  age_out_buffer aob(10);

  MPI_Barrier(MPI_COMM_WORLD);
  REQUIRE(aob.oldest_entry()==9);
  MPI_Barrier(MPI_COMM_WORLD);
  aob.promote_to_top(8);
  MPI_Barrier(MPI_COMM_WORLD);
  REQUIRE(aob.oldest_entry()==9);
  MPI_Barrier(MPI_COMM_WORLD);
  aob.promote_to_top(9);
  MPI_Barrier(MPI_COMM_WORLD);
  REQUIRE(aob.oldest_entry()==7);
  MPI_Barrier(MPI_COMM_WORLD);
  aob.promote_to_top(6);
  MPI_Barrier(MPI_COMM_WORLD);
  REQUIRE(aob.oldest_entry()==7);
  MPI_Barrier(MPI_COMM_WORLD);
  aob.promote_to_top(7);
  MPI_Barrier(MPI_COMM_WORLD);
  REQUIRE(aob.oldest_entry()==5);
  MPI_Barrier(MPI_COMM_WORLD);
  aob.promote_to_top(9);
  MPI_Barrier(MPI_COMM_WORLD);
  REQUIRE(aob.oldest_entry()==5);
  MPI_Barrier(MPI_COMM_WORLD);
  
}
TEST_CASE("repeat","[ReadingSI]") {
  age_out_buffer aob(10);

  aob.promote_to_top(8);
  aob.promote_to_top(8);
  aob.promote_to_top(8);
  aob.promote_to_top(8);

  REQUIRE(aob.oldest_entry()==9);

}
