#pragma once
#include<string>
#include<Eigen/Dense>
#include<hdf5.h>

class chunk_reader{
  public:
    chunk_reader(){
      validate_threadsafety();
    }
    chunk_reader(const std::string &basepath, int number_of_keys, int naux, int nao):
     basepath_(basepath),
     number_of_keys_(number_of_keys),
     element_size_(naux*nao*nao*2),
     naux_(naux),
     nao_(nao){
      validate_threadsafety();
      parse_meta();
    }
    const Eigen::VectorXi &chunk_indices() const{ return chunk_indices_;}
    int number_of_keys() const{ return number_of_keys_;}
    int element_size() const{return element_size_;}
    void validate_threadsafety(){
      hbool_t is_ts;
      H5is_library_threadsafe (&is_ts);
      if(!is_ts) throw std::runtime_error("this hdf5 library is not threadsafe. Threadsafety is a prerequisite for accessing files multiple times");
    }

    //read key ('chunk') into buffer
    void read_key(int key, double *buffer);
  private:
    void parse_meta();
    //figure out where we stored the data (due to manual chunking it may be distributed)
    void find_file_and_offset(int key, std::string &path, int &chunk_name, unsigned long long &offset);
    void read_key_at_offset(const std::string &filepath, int chunk_name, unsigned long long offset, double *buffer);
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
