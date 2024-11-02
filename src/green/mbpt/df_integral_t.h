/*
 * Copyright (c) 2023 University of Michigan.
 *
 */

#ifndef GREEN_DFINTEGRAL_H
#define GREEN_DFINTEGRAL_H

#include <green/symmetry/symmetry.h>
#include <green/utils/mpi_shared.h>
#include <green/utils/mpi_utils.h>

#include <green/mbpt/except.h>

#include "df_legacy_reader.h"
#include "df_buffered_reader.h"

namespace green::mbpt {
  enum integral_symmetry_type_e { direct, conjugated, transposed };
  /**
   * @brief Integral class to parse Density fitted 3-center integrals, handles reading given by the path argument
   */
  class df_integral_t {
    // prefixes for hdf5
    const std::string _chunk_basename    = "VQ";
    const std::string _corr_path         = "df_ewald.h5";
    const std::string _corr_basename     = "EW";
    const std::string _corr_bar_basename = "EW_bar";

    using bz_utils_t                     = symmetry::brillouin_zone_utils<symmetry::inv_symm_op>;
    using int_data                       = utils::shared_object<ztensor<4>>;

  public:
    df_integral_t(const std::string& path, int nao, int NQ, const bz_utils_t& bz_utils) :
      _base_path(path),
      _number_of_keys(bz_utils.symmetry().num_kpair_stored()),
      _vij_Q(path, nao, NQ), //initialize legacy reader
      _vij_Q_buffer(path, nao, NQ, _number_of_keys), //initialize buffered reader
        _k0(-1), _NQ(NQ), _bz_utils(bz_utils) {
    }

    virtual ~df_integral_t() {}

    void read_integrals(size_t k1, size_t k2){
      _vij_Q.read_integrals(momenta_to_symmred_key(k1,k2));
    }

    void Complex_DoubleToType(const std::complex<double>* in, std::complex<double>* out, size_t size) {
      memcpy(out, in, size * sizeof(std::complex<double>));
    }

    void Complex_DoubleToType(const std::complex<double>* in, std::complex<float>* out, size_t size) {
      for (int i = 0; i < size; ++i) {
        out[i] = static_cast<std::complex<float>>(in[i]);
      }
    }

    /**
     * read next part of the G=0 correction to interaction integral for the specific k-point
     * @param file - file to be used
     * @param k - k-point
     */
    void read_correction(int k) {
      auto shape = _vij_Q.shape();
      _v0ij_Q.resize(shape[1], shape[2], shape[3]);
      _v_bar_ij_Q.resize(shape[1], shape[2], shape[3]);
      // avoid unnecessary reading
      if (k == _k0) {
        // we have data cached
        return;
      }
      _k0                 = k;
      std::string   fname = _base_path + "/" + _corr_path;
      h5pp::archive ar(fname);
      // Construct integral dataset name
      std::string   dsetnum = _corr_basename + "/" + std::to_string(k);
      // read data
      ar[dsetnum] >> reinterpret_cast<double*>(_v0ij_Q.data());
      // Construct integral dataset name
      dsetnum = _corr_bar_basename + "/" + std::to_string(k);
      // read data
      ar[dsetnum] >> reinterpret_cast<double*>(_v_bar_ij_Q.data());
      ar.close();
    };

    /**
     * Determine the type of symmetries for the integral based on the current k-points
     *
     * @param k1 incomming k-point
     * @param k2 outgoing k-point
     * @return A pair of sign and type of applied symmetry
     */
    std::pair<int, integral_symmetry_type_e> v_type(size_t k1, size_t k2) {
      size_t idx  = momenta_to_key(k1,k2);
      // determine sign
      int    sign = (k1 >= k2) ? 1 : -1;
      // determine applied symmetry type
      // by default no symmetries applied
      integral_symmetry_type_e symmetry_type = direct;
      if (_bz_utils.symmetry().conj_kpair_list()[idx] != idx) {
        symmetry_type = conjugated;
      } else if (_bz_utils.symmetry().trans_kpair_list()[idx] != idx) {
        symmetry_type = transposed;
      }
      return std::make_pair(sign, symmetry_type);
    }

    /**
     * Extract V(Q, i, j) with given (k1, k2) from chunks of integrals (_vij_Q)
     * Note that Q here denotes the auxiliary basis index, not the transfer momentum
     * Also apply conjugate transpose, conjugate, or transpose.
     * @tparam prec
     * @param vij_Q_k1k2
     * @param k1
     * @param k2
     */
    template <typename prec>
    void symmetrize(tensor<prec, 3>& vij_Q_k1k2, size_t k1, size_t k2, size_t NQ_offset = 0, size_t NQ_local = 0) {
      std::pair<int, integral_symmetry_type_e> vtype     = v_type(k1, k2);
      int                                      NQ        = _NQ;
      NQ_local                                           = (NQ_local == 0) ? NQ : NQ_local;
      int nao=_vij_Q.shape()[2]; 
      typedef Eigen::Map<const Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> map_t;
      if (vtype.first < 0) {
        for (int Q = NQ_offset, Q_loc = 0; Q_loc < NQ_local; ++Q, ++Q_loc) {
          int key=momenta_to_symmred_key(k1,k2);
          map_t vij_map(_vij_Q(key, Q),nao,nao);
          matrix(vij_Q_k1k2(Q_loc)) = vij_map.transpose().conjugate().cast<prec>();
        }
      } else {
        for (int Q = NQ_offset, Q_loc = 0; Q_loc < NQ_local; ++Q, ++Q_loc) {
          int key=momenta_to_symmred_key(k1,k2);
          map_t vij_map(_vij_Q(key, Q),nao,nao);
          matrix(vij_Q_k1k2(Q_loc)) = vij_map.cast<prec>();
        }
      }
      if (vtype.second == conjugated) {  // conjugate 
        for (int Q = NQ_offset, Q_loc = 0; Q_loc < NQ_local; ++Q, ++Q_loc) {
          matrix(vij_Q_k1k2(Q_loc)) = matrix(vij_Q_k1k2(Q_loc)).conjugate();

        }
      } else if (vtype.second == transposed) {  // transpose
        for (int Q = NQ_offset, Q_loc = 0; Q_loc < NQ_local; ++Q, ++Q_loc) {
          matrix(vij_Q_k1k2(Q_loc)) = matrix(vij_Q_k1k2(Q_loc)).transpose().eval();
        }
      }
    }

    //const ztensor<4>& vij_Q() const { return _vij_Q()->object(); }
    const ztensor<3>& v0ij_Q() const { return _v0ij_Q; }
    const ztensor<3>& v_bar_ij_Q() const { return _v_bar_ij_Q; }
    const std::complex<double> *vij_Q(int k1, int k2) const 
    { 
      return _vij_Q(momenta_to_symmred_key(k1,k2)); 
    } 

    int momenta_to_key(int k1, int k2) const{
      size_t idx = (k1 >= k2) ? k1 * (k1 + 1) / 2 + k2 : k2 * (k2 + 1) / 2 + k1;  // k-pair = (k1, k2) or (k2, k1)
      return idx;
    }
    int momenta_to_symmred_key(int k1, int k2) const{
      int idx=momenta_to_key(k1,k2);
      // determine type
      if (_bz_utils.symmetry().conj_kpair_list()[idx] != idx) {
        idx = _bz_utils.symmetry().conj_kpair_list()[idx];
      } else if (_bz_utils.symmetry().trans_kpair_list()[idx] != idx) {
        idx = _bz_utils.symmetry().trans_kpair_list()[idx];
      }
      int idx_red = _bz_utils.symmetry().irre_pos_kpair(idx);
      return idx_red;
    }

    void reset() {
      _vij_Q.reset();
    }

  private:
    int                       _number_of_keys;
    df_legacy_reader _vij_Q;
    df_buffered_reader _vij_Q_buffer;
    // G=0 correction to coulomb integral stored in density fitting format for second-order e3xchange diagram
    ztensor<3>                _v0ij_Q;
    ztensor<3>                _v_bar_ij_Q;

    bool                      _exch;
    // current leading index
    int                       _k0;
    long                      _NQ;
    const bz_utils_t&         _bz_utils;

    // base path to integral files
    std::string               _base_path;
  };

}  // namespace green::mbpt

#endif  // GF2_DFINTEGRAL_H
