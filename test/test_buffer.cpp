#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "buffer.hpp"
#include <mpi.h>

TEST_CASE("Init","[buffer]") {
  std::size_t element_size=100;
  std::size_t total_keys=100;
  reader R;
  buffer b(element_size, total_keys, total_keys, &R);
}
TEST_CASE("ValSize","[buffer]") {
  int nao=26;
  int naux=200;
  int nKQ=1200; //#K x #Q, total number of keys
  int element_size=nao*nao*naux*2; //the 2 is for complex
  reader R;
  buffer b(element_size, nKQ, nKQ, &R);
  REQUIRE(b.element_size()==element_size);
  REQUIRE(b.number_of_keys()== nKQ);
}
TEST_CASE("NelemHeuristics","[buffer]") {
  int nao=26;
  int naux=200;
  int nKQ=12000000; //#K x #Q, total number of keys. This is a lot here.
  int element_size=nao*nao*naux*2; //the 2 is for complex

  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //find out how many elements we could in principle fit into a buffer
  int all=buffer::n_buffer_elem_heuristics(1., element_size, nKQ);
  int half=buffer::n_buffer_elem_heuristics(0.5, element_size, nKQ);
  int quarter=buffer::n_buffer_elem_heuristics(0.25, element_size, nKQ);

  REQUIRE_THAT(all, Catch::Matchers::WithinAbs( 2.*half, 2.));
  REQUIRE_THAT(all, Catch::Matchers::WithinAbs(4.*quarter, 2.));

  //this is a case where all entries fit -- just load them all.
  nKQ=1200;
  reader R;
  buffer b2(element_size, buffer::n_buffer_elem_heuristics(0.5, element_size, nKQ), nKQ, &R);
  REQUIRE(buffer::n_buffer_elem_heuristics(0.5, element_size, nKQ)==b2.number_of_keys());
}
TEST_CASE("InitialStatus","[buffer]") {

  int nao=26;
  int naux=200;
  int nKQ=1200; 
  int element_size=nao*nao*naux*2;
  reader R;

  buffer b(element_size, nKQ, nKQ, &R);
  for(int i=0;i<nKQ;++i){
    REQUIRE(b.element_status(i)== status_elem_unavailable);
  }
}
