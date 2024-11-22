/*
 * Copyright (c) 2023 University of Michigan.
 *
 */

#ifndef GREEN_DFBR_H
#define GREEN_DFBR_H

#include <green/symmetry/symmetry.h>
#include <green/utils/mpi_shared.h>
#include <green/utils/mpi_utils.h>

#include <green/mbpt/except.h>

#include"buffered_reader/chunk_reader.hpp"
#include"buffered_reader/buffer.hpp"

namespace green::mbpt {
  class df_buffered_reader{
  /**
   * @brief Integral class to read 3-central integrals using buffered reading
   * @param path: path of hdf5 directory
   * @param nao: number of atomic orbitals
   * @param NQ: number of auxiliary orbitals
   * @param number_of_keys: total number of elements in hdf5 files (each of size nao**2*NQ
   * @param buffer_mem_ratio: parameter for heuristics of how much of total physical memory should be allocated for a buffer
   */

    const std::string _chunk_basename    = "VQ";

  public:
    df_buffered_reader(const std::string& path, int nao, int NQ, int number_of_keys, double buffer_mem_ratio=0.5) :
        _base_path(path), _k0(-1), _NQ(NQ), _nao(nao),
        _number_of_keys(number_of_keys),
        _number_of_buffered_elements(buffer::n_buffer_elem_heuristics(buffer_mem_ratio, nao*nao*NQ*sizeof(std::complex<double>), number_of_keys)),
        _reader(path, number_of_keys, NQ, nao),
        _buffer(nao*nao*NQ*2, number_of_keys, _number_of_buffered_elements, &_reader, true, false){ //'*2' for double storage inside buffer. also 'true' for single thread read which otherwise causes problems if multiple instances want to read.
      h5pp::archive ar(path + "/meta.h5");
      if(ar.has_attribute("__green_version__")) {
        std::string int_version = ar.get_attribute<std::string>("__green_version__");
        if (int_version.rfind(INPUT_VERSION, 0) != 0) {
          throw mbpt_outdated_input("Integral files at '" + path +"' are outdated, please run migration script python/migrate.py");
        }
      } else {
        throw mbpt_outdated_input("Integral files at '" + path +"' are outdated, please run migration script python/migrate.py");
      }
      ar.close();

      //manually set shape as nkey*nQ*nao*nao
      _shape[0]=number_of_keys; _shape[1]=NQ; _shape[2]=nao; _shape[3]=_nao;
    }

    virtual ~df_buffered_reader() {}

    void read_integrals(size_t idx_red) {
      //noop. This would only be needed for the legacy reader
      ;
    }

    void reset() {
      //nothing to reset
      ;
    }

    const std::complex<double> *access_element(int key){ 
      return (const std::complex<double>*)_buffer.access_element(key);
    }

    void release_element(int key){ 
      _buffer.release_element(key);
    }

    const std::array<size_t, 4> shape() const{ 
      return _shape;
    }

    const long nao() const{ return _nao;}


  private:

    // current leading index
    int                       _k0;
    long                      _current_chunk;
    long                      _chunk_size;
    long                      _NQ;
    long                      _nao;

    // base path to integral files
    std::string               _base_path;
    int _number_of_keys;
    int _number_of_buffered_elements;
    chunk_reader              _reader;
    buffer                    _buffer;
    std::array<size_t, 4>     _shape;
  };

}  // namespace green::mbpt

#endif  // GF2_DFBR_H
