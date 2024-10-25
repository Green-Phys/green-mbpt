#pragma once
#include<string>
#include<Eigen/Dense>
#include<hdf5.h>
#include"reader.hpp"

class chunk_reader:public reader{
  public:
    chunk_reader(){
      validate_threadsafety();
    }
    chunk_reader(const std::string &basepath, int number_of_keys, int naux, int nao):
     reader(basepath, number_of_keys, naux, nao)
    {
      validate_threadsafety();
      parse_meta();
    }
    void validate_threadsafety(){
      hbool_t is_ts;
      H5is_library_threadsafe (&is_ts);
      if(!is_ts) throw std::runtime_error("this hdf5 library is not threadsafe. Threadsafety is a prerequisite for accessing files multiple times");
    }

    //read key ('chunk') into buffer
    virtual void read_key(int key, double *buffer);
  private:
    void parse_meta();
    //figure out where we stored the data (due to manual chunking it may be distributed)
    void find_file_and_offset(int key, std::string &path, int &chunk_name, unsigned long long &offset);
    void read_key_at_offset(const std::string &filepath, int chunk_name, unsigned long long offset, double *buffer);
};
