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

  H5Sclose(dataspace_id);
  H5Dclose(dataset_id);
  H5Fclose(file_id);
}
void chunk_reader::read_key(int key, double *buffer){
  std::string filepath;
  unsigned long long offset;
  int chunk_name;

  find_file_and_offset(key, filepath, chunk_name, offset);

  read_key_at_offset(filepath, chunk_name, offset, buffer);
}
void chunk_reader::find_file_and_offset(int key, std::string &filepath, int &chunk_name, unsigned long long &offset){
  auto chunk_ptr=std::lower_bound ((chunk_indices_.begin()), chunk_indices_.end(), key);
  if(*chunk_ptr != key) chunk_ptr--;
  chunk_name=*chunk_ptr;
  offset=key-chunk_name;
  std::stringstream filepath_sstr; filepath_sstr<<basepath_<<"/VQ_"<<chunk_name<<".h5";
  filepath=filepath_sstr.str();
  //std::cout<<"chunk number is: "<<chunk_name<<" file path: "<<filepath<<" offset: "<<offset<<std::endl;
}
void chunk_reader::read_key_at_offset(const std::string &filepath, int chunk_name, unsigned long long offset, double *buffer){
    hid_t file_id, dataset_id, dataspace_id, memspace_id;
    int rank=4;
    hsize_t start[4] = {offset, 0    , 0   , 0     }; // Starting index for the slice
    hsize_t count[4] = {1     , naux_, nao_, 2*nao_}; // Number of elements to read along each dimension

    file_id = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
      throw std::runtime_error("Error opening file: "+filepath);
    }

    std::stringstream hdf5_path; hdf5_path<<"/"<<chunk_name;   
    dataset_id = H5Dopen(file_id, hdf5_path.str().c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
      throw std::runtime_error("Error opening dataset: "+hdf5_path.str()+" for file: "+filepath);
    }

    dataspace_id = H5Dget_space(dataset_id);

    herr_t status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, start, NULL, count, NULL);
    if (status < 0) {
      throw std::runtime_error("Error selecting hyperslab: "+hdf5_path.str()+" for file: "+filepath);
    }

    memspace_id = H5Screate_simple(rank, count, NULL);
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, H5P_DEFAULT, buffer);
    if (status < 0) {
      throw std::runtime_error("Error reading data: "+hdf5_path.str()+" for file: "+filepath);
    }

    H5Sclose(memspace_id);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
}
