#include <catch2/catch_test_macros.hpp>
#include "buffer.hpp"
#include "chunk_reader.hpp"
#include <mpi.h>
#include <openssl/sha.h>
#include <iomanip>


TEST_CASE("HashOfKeys","[ReadingSI]") {
  int chunks_per_file=336;
  int total_files=36;
  int nao=26;
  int naux=200;
  int number_of_keys=chunks_per_file*total_files;

  unsigned char digest[SHA_DIGEST_LENGTH];


  chunk_reader c(HDF5_DATA_DIR, number_of_keys, naux, nao); //test these numbers
  buffer b(c.element_size(), number_of_keys, 100, &c, false, false);
  if(b.shmem_rank()==0){
  for(int i=0;i<5;++i){
    const double* val=b.access_element(i);
    SHA1((unsigned char*)val, c.element_size()*sizeof(double), digest);
    std::cout<<"digest of: "<<i<<" is: "<<std::hex<<std::setw(2)<<std::setfill('0');
    for (int i = 0; i < SHA_DIGEST_LENGTH; i++) {
        std::cout << (int)digest[i];
    }
    std::cout<<std::dec<<std::endl;

    b.release_element(i);
  }

  }
  MPI_Barrier(MPI_COMM_WORLD);
}

