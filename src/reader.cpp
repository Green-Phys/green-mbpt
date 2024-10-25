#include"reader.hpp"
//we do not actually read at all, we just sleep.
void reader::read_key(int key, double *buffer){
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  for(int i=0;i<element_size_;++i){
    buffer[i]=42;
  }
}
