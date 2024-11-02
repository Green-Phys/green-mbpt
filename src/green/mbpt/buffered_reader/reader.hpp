#pragma once
#include <chrono>
#include <thread>
#include<Eigen/Dense>
#include<string>

class reader{
public:
  reader(){}
  reader(const std::string &basepath, int number_of_keys, int naux, int nao):
     basepath_(basepath),
     number_of_keys_(number_of_keys),
     element_size_(naux*nao*nao*2),
     naux_(naux),
     nao_(nao){
    }
    const Eigen::VectorXi &chunk_indices() const{ return chunk_indices_;}
    int number_of_keys() const{ return number_of_keys_;}
    int element_size() const{return element_size_;}

    //read key ('chunk') into buffer
    virtual void read_key(int key, double *buffer);
protected:
    const std::string basepath_;
    Eigen::VectorXi chunk_indices_;
    unsigned long long number_of_keys_;

    //total size of a chunk: naux*nao*nao*2
    unsigned long long element_size_;
    //number of aux orbitals, dimension1 in hdf5
    unsigned long long naux_;
    //number of ao, dimension2 and dimension3(/2) in hdf5 
    unsigned long long nao_;
};
