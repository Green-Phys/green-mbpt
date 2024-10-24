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
  chunk_reader c(HDF5_DATA_DIR, 12096, 2163206); //test these numbers

  EXPECT_EQ(c.chunk_indices()[0],0);
  EXPECT_EQ(c.chunk_indices()[1],336);
  EXPECT_EQ(c.chunk_indices()[35],11760);
}
