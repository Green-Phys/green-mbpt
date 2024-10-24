#include"chunk_reader.hpp"
#include <hdf5.h>
#include <Eigen/Dense>
#include <iostream>

void chunk_reader::parse_meta(){
  std::string filename=basepath_+"/meta.h5";

  hid_t file_id; file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    throw std::runtime_error("Error opening HDF5 file: "+filename);
  }

  std::string datasetname="/chunk_indices";
  hid_t dataset_id = H5Dopen2(file_id, datasetname.c_str(), H5P_DEFAULT);
  if (dataset_id < 0) {
    throw("Error opening dataset: "+ datasetname);
  }

  hid_t dataspace_id = H5Dget_space(dataset_id); 
  hsize_t dims[1]; 
  int ndims = H5Sget_simple_extent_dims(dataspace_id, dims, NULL);

  chunk_indices_.resize(dims[0]);
  herr_t status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(chunk_indices_[0])); 
  if (status < 0) { throw("Error reading chunk information.\n");}
}
