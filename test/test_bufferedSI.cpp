#include "gtest/gtest.h"
#include "buffer.hpp"
#include "chunk_reader.hpp"
#include <mpi.h>

TEST(ReadingSI, Init) {
  int chunks_per_file=336;
  int total_files=36;
  int nao=26;
  int naux=200;
  int number_of_keys=chunks_per_file*total_files;

  chunk_reader c(HDF5_DATA_DIR, number_of_keys, naux, nao); //test these numbers
  buffer b(c.element_size(), number_of_keys, number_of_keys, &c);

  const double* val=b.access_element(0);
  EXPECT_NEAR(val[0], 5.26945, 1.e-5);
  b.release_element(0);
}
TEST(ReadingSI, ReadAllInts) {
  int chunks_per_file=336;
  int total_files=36;
  int nao=26;
  int naux=200;
  int number_of_keys=chunks_per_file*total_files;

  chunk_reader c(HDF5_DATA_DIR, number_of_keys, naux, nao); //test these numbers
  buffer b(c.element_size(), number_of_keys, number_of_keys, &c);

  for(int i=b.shmem_rank();i<chunks_per_file*total_files;i+=b.shmem_size()){
    if(i>=chunks_per_file*total_files) break;
    const double* val=b.access_element(i);
  }
  for(int i=b.shmem_rank();i<chunks_per_file*total_files;i+=b.shmem_size()){
    if(i>=chunks_per_file*total_files) break;
    b.release_element(i);
  }
}




