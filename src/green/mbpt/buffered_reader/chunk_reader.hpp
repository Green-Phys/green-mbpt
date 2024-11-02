#pragma once
#include<string>
#include<Eigen/Dense>
#include<hdf5.h>
#include<iostream>
#include"reader.hpp"

class chunk_reader:public reader{
  public:
    chunk_reader(bool verbose=false):
     elapsed_(0){
      validate_threadsafety();
      ctr_=0;
      verbose_=verbose;
    }
    chunk_reader(const std::string &basepath, int number_of_keys, int naux, int nao, bool verbose=false):
     reader(basepath, number_of_keys, naux, nao),
     elapsed_(0)
    {
      ctr_=0;
      verbose_=verbose;
      validate_threadsafety();
      parse_meta();
    }
    ~chunk_reader(){
      if(verbose_){
        double av_read_time=elapsed_.count()/ctr_;
        double read_size=element_size_*sizeof(double);
        std::cout<<"total reads: "<<ctr_<<" effective read rate: "<<read_size/av_read_time/1024./1024.<<" MB/s"<<std::endl;
      }
    }
    void validate_threadsafety(){
      hbool_t is_ts;
      H5is_library_threadsafe (&is_ts);
      //if(!is_ts) throw std::runtime_error("this hdf5 library is not threadsafe. Threadsafety is a prerequisite for accessing files multiple times");
    }

    //read key ('chunk') into buffer
    virtual void read_key(int key, double *buffer);
  private:
    void parse_meta();
    //figure out where we stored the data (due to manual chunking it may be distributed)
    void find_file_and_offset(int key, std::string &path, int &chunk_name, unsigned long long &offset);
    void read_key_at_offset(const std::string &filepath, int chunk_name, unsigned long long offset, double *buffer);

    std::chrono::duration<double> elapsed_;
    std::size_t ctr_;
    bool verbose_;
};
