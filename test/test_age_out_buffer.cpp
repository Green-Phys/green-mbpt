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

  EXPECT_EQ(aob.oldest_entry(), 9);
  aob.promote_to_top(8);
  EXPECT_EQ(aob.oldest_entry(), 9);
  aob.promote_to_top(9);
  EXPECT_EQ(aob.oldest_entry(), 7);
  aob.promote_to_top(6);
  EXPECT_EQ(aob.oldest_entry(), 7);
  aob.promote_to_top(7);
  EXPECT_EQ(aob.oldest_entry(), 5);
  aob.promote_to_top(9);
  EXPECT_EQ(aob.oldest_entry(), 5);
  
}
TEST(age_out_buffer, repeat) {
  age_out_buffer aob(10);

  aob.promote_to_top(8);
  aob.promote_to_top(8);
  aob.promote_to_top(8);
  aob.promote_to_top(8);

  EXPECT_EQ(aob.oldest_entry(), 9);

}
