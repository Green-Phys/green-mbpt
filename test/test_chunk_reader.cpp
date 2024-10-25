#include "gtest/gtest.h"
#include "chunk_reader.hpp"
#include <mpi.h>
#include <chrono>
#include <thread>

TEST(chunk_reader, Init) {
  chunk_reader c;
}
//these tests are straight from the Si 6x6x6 example
TEST(chunk_reader, InitBasepath) {

  int chunks_per_file=336;
  int total_files=36;
  int nao=26;
  int naux=200;
  int number_of_keys=chunks_per_file*total_files;

  chunk_reader c(HDF5_DATA_DIR, number_of_keys, naux, nao); //test these numbers

  EXPECT_EQ(c.chunk_indices()[0],0);
  EXPECT_EQ(c.chunk_indices()[1],336);
  EXPECT_EQ(c.chunk_indices()[35],11760);
}
TEST(chunk_reader, ReadSomething){
  int chunks_per_file=336;
  int total_files=36;
  int nao=26;
  int naux=200;
  int number_of_keys=chunks_per_file*total_files;

  chunk_reader c(HDF5_DATA_DIR, number_of_keys, naux, nao); //test these numbers

  Eigen::VectorXd data(c.element_size());
  c.read_key(0, &(data[0]));
  EXPECT_NEAR(data[0], 5.26945, 1.e-5);
  c.read_key(335, &(data[0]));
  c.read_key(336, &(data[0]));
  c.read_key(2000, &(data[0]));
}
