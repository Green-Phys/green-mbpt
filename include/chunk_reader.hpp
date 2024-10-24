#pragma once
#include<string>
#include<Eigen/Dense>

class chunk_reader{
  public:
    chunk_reader(){}
    chunk_reader(const std::string &basepath, int number_of_keys, int element_size):
     basepath_(basepath),
     number_of_keys_(number_of_keys),
     element_size_(element_size){
      parse_meta();
    }
    const Eigen::VectorXi &chunk_indices() const{ return chunk_indices_;}
    int number_of_keys() const{ return number_of_keys_;}
    int element_size() const{return element_size_;}
  private:
    void parse_meta();
    const std::string basepath_;
    Eigen::VectorXi chunk_indices_;
    int number_of_keys_;
    int element_size_;
};
