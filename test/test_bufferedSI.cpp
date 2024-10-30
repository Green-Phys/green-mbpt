#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "buffer.hpp"
#include "chunk_reader.hpp"
#include <mpi.h>

TEST_CASE("Init","[ReadingSI]") {

  int chunks_per_file=336;
  int total_files=36;
  int nao=26;
  int naux=200;
  int number_of_keys=chunks_per_file*total_files;

  chunk_reader c(HDF5_DATA_DIR, number_of_keys, naux, nao); //test these numbers
  buffer b(c.element_size(), number_of_keys, number_of_keys, &c);

  const double* val=b.access_element(0);
  REQUIRE_THAT(val[0], Catch::Matchers::WithinAbs(5.26945, 1.e-5));
  b.release_element(0);
}
TEST_CASE("ReadAllIntsConsecutively","[ReadingSI]") {
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

TEST_CASE("ReadAllIntsSmallBuffer","[ReadingSI]") {
  int chunks_per_file=336;
  int total_files=36;
  int nao=26;
  int naux=200;
  int number_of_keys=chunks_per_file*total_files;
  int n_buffered_elem=100;

  chunk_reader c(HDF5_DATA_DIR, number_of_keys, naux, nao); //test these numbers
  buffer b(c.element_size(), number_of_keys, n_buffered_elem, &c);

  for(int i=b.shmem_rank();i<chunks_per_file*total_files;i+=b.shmem_size()){
    if(i>=chunks_per_file*total_files) break;
    //std::cout<<"rank: "<<b.shmem_rank()<<" accessing element: "<<i<<std::endl;
    const double* val=b.access_element(i);
    b.release_element(i);
  }
}
TEST_CASE("ReadAllIntsConsecutivelyLargeStride","[ReadingSI]") {
  int chunks_per_file=336;
  int total_files=36;
  int nao=26;
  int naux=200;
  int number_of_keys=chunks_per_file*total_files;

  chunk_reader c(HDF5_DATA_DIR, number_of_keys, naux, nao); //test these numbers
  buffer b(c.element_size(), number_of_keys, number_of_keys, &c, false, false);

  int stride=number_of_keys/b.shmem_size();
  int start=number_of_keys/b.shmem_size()*b.shmem_rank();
  int stop=std::min(number_of_keys/b.shmem_size()*(b.shmem_rank()+1), number_of_keys);
  for(int i=start;i<stop;++i){
    //std::cout<<"rank: "<<b.shmem_rank()<<" reading: "<<i<<std::endl;
    const double* val=b.access_element(i);
    b.release_element(i);
  }
}
