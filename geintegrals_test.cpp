
#include "green/integrals/buffered_reader/buffer.hpp"
#include "green/integrals/buffered_reader/chunk_reader.hpp"
#include <string>
#include <mpi.h>
#include <filesystem>

int main(int argc, char** argv) {

  // initialize mpi
  MPI_Init(&argc, &argv);
  int size = 0;
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  //// Germanium 2x2x2 data
  //int chunks_per_file=36;
  //int total_files=1;
  //int nao=54;
  //int naux=342;
  //int number_of_keys=chunks_per_file*total_files;
  //std::string TEST_PATH = "/pauli-storage/gharsha/Germanium/gw_2x2x2/df_hf_int";

  // Germanium 6x6x6 data
  int chunks_per_file=2357;
  int total_files=6;
  int nao=54;
  int naux=342;
  int number_of_keys=chunks_per_file*total_files;
  std::string TEST_PATH = "/pauli-storage/gharsha/Germanium/gw_6x6x6/df_hf_int";

  if (!rank) {
    std::cout << "file system address:" << std::endl;
    std::cout << TEST_PATH << std::endl;
  }
  if(!std::filesystem::exists(TEST_PATH)){ std::cerr<<"hdf5 data not found. aborting test"<<std::endl; return 1;}
  chunk_reader c(TEST_PATH, number_of_keys, naux, nao, true); //test these numbers
  if (!rank) {
    std::cout << "chunk reader initialized" << std::endl;
  }
  buffer b(c.element_size(), number_of_keys, 1000, &c);
  const double* val=b.access_element(0);
  if (!rank) {
    std::cout << "check value 0: " << val[0] << std::endl;
  }
  //REQUIRE_THAT(val[0], Catch::Matchers::WithinAbs(5.26945, 1.e-5));
  b.release_element(0);

  MPI_Barrier(MPI_COMM_WORLD);
  //MPI_Finalize();
  return 0;
}
