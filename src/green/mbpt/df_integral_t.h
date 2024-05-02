/*
 * Copyright (c) 2023 University of Michigan.
 *
 */

#ifndef GREEN_DFINTEGRAL_H
#define GREEN_DFINTEGRAL_H

#include <green/symmetry/symmetry.h>
#include <green/utils/mpi_shared.h>
#include <green/utils/mpi_utils.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>

namespace green::mbpt {
  enum integral_symmetry_type_e { direct, conjugated, transposed };
  /**
   * @brief Integral class read Density fitted 3-center integrals from a HDF5 file, given by the path argument
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
    using MatrixXcd = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixXcf = Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixXd  = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    df_integral_t(const std::string& path, int nao, int NQ, const bz_utils_t& bz_utils) :
        _base_path(path), _k0(-1), _current_chunk(-1), _chunk_size(0), _NQ(NQ), _bz_utils(bz_utils) {
      h5pp::archive ar(path + "/meta.h5");
      hid_t         file = H5Fopen((path + "/meta.h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      ar["chunk_size"] >> _chunk_size;
      ar.close();
      _vij_Q = std::make_shared<int_data>(_chunk_size, NQ, nao, nao);
    }

    virtual ~df_integral_t() {}

    /**
     * Read next part of the interaction integral from
     * @param k1
     * @param k2
     * @param type
     */
    void read_integrals(size_t k1, size_t k2) {
      assert(k1 >= 0);
      assert(k2 >= 0);
      // Find corresponding index for k-pair (k1,k2). Only k-pair with k1 > k2 will be stored.
      size_t idx = (k1 >= k2) ? k1 * (k1 + 1) / 2 + k2 : k2 * (k2 + 1) / 2 + k1;  // k-pair = (k1, k2) or (k2, k1)
      // Corresponding symmetry-related k-pair
      if (_bz_utils.symmetry().conj_kpair_list()[idx] != idx) {
        idx = _bz_utils.symmetry().conj_kpair_list()[idx];
      } else if (_bz_utils.symmetry().trans_kpair_list()[idx] != idx) {
        idx = _bz_utils.symmetry().trans_kpair_list()[idx];
      }
      long idx_red = _bz_utils.symmetry().irre_pos_kpair(idx);
      if ((idx_red / _chunk_size) == _current_chunk) return;  // we have data cached

      _current_chunk = idx_red / _chunk_size;

      size_t c_id    = _current_chunk * _chunk_size;
      (*_vij_Q).fence();
      if (!utils::context.node_rank) read_a_chunk(c_id, _vij_Q->object());
      (*_vij_Q).fence();
    }

    void read_a_chunk(size_t c_id, ztensor<4>& V_buffer) {
      std::string   fname = _base_path + "/" + _chunk_basename + "_" + std::to_string(c_id) + ".h5";
      hid_t         file  = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      h5pp::archive ar(fname);
      ar["/" + std::to_string(c_id)] >> reinterpret_cast<double*>(V_buffer.data());
      ar.close();
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
      auto shape = _vij_Q->object().shape();
      _v0ij_Q.resize(shape[1], shape[2], shape[3]);
      _v_bar_ij_Q.resize(shape[1], shape[2], shape[3]);
      // avoid unnecessary reading
      if (k == _k0) {
        // we have data cached
        return;
      }
      _k0                 = k;
      std::string   inner = std::to_string(_current_chunk * _chunk_size);
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
      size_t idx  = (k1 >= k2) ? k1 * (k1 + 1) / 2 + k2 : k2 * (k2 + 1) / 2 + k1;  // k-pair = (k1, k2) or (k2, k1)
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
     * @tparam prec
     * @param vij_Q_k1k2
     * @param k1
     * @param k2
     */
    template <typename prec>
    void symmetrize(tensor<prec, 3>& vij_Q_k1k2, size_t k1, size_t k2, size_t NQ_offset = 0, size_t NQ_local = 0) {
      int                                      k1k2_wrap = wrap(k1, k2);
      std::pair<int, integral_symmetry_type_e> vtype     = v_type(k1, k2);
      int                                      NQ        = _NQ;
      NQ_local                                           = (NQ_local == 0) ? NQ : NQ_local;
      auto& vij_Q                                        = _vij_Q->object();
      if (vtype.first < 0) {
        for (int Q = NQ_offset, Q_loc = 0; Q_loc < NQ_local; ++Q, ++Q_loc) {
          matrix(vij_Q_k1k2(Q_loc)) = matrix(vij_Q(k1k2_wrap, Q)).transpose().conjugate().cast<prec>();
        }
      } else {
        for (int Q = NQ_offset, Q_loc = 0; Q_loc < NQ_local; ++Q, ++Q_loc) {
          matrix(vij_Q_k1k2(Q_loc)) = matrix(vij_Q(k1k2_wrap, Q)).cast<prec>();
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

    const ztensor<4>& vij_Q() const { return _vij_Q->object(); }
    const ztensor<3>& v0ij_Q() const { return _v0ij_Q; }
    const ztensor<3>& v_bar_ij_Q() const { return _v_bar_ij_Q; }

    int               wrap(int k1, int k2) {
      size_t idx = (k1 >= k2) ? k1 * (k1 + 1) / 2 + k2 : k2 * (k2 + 1) / 2 + k1;  // k-pair = (k1, k2) or (k2, k1)
      // determine type
      if (_bz_utils.symmetry().conj_kpair_list()[idx] != idx) {
        idx = _bz_utils.symmetry().conj_kpair_list()[idx];
      } else if (_bz_utils.symmetry().trans_kpair_list()[idx] != idx) {
        idx = _bz_utils.symmetry().trans_kpair_list()[idx];
      }
      int idx_red = _bz_utils.symmetry().irre_pos_kpair(idx);
      return idx_red % _chunk_size;
    }

    void reset() {
      _current_chunk = -1;
      _k0            = -1;
    }

  private:
    // Coulomb integrals stored in density fitting format
    std::shared_ptr<int_data> _vij_Q;
    // G=0 correction to coulomb integral stored in density fitting format for second-order e3xchange diagram
    ztensor<3>                _v0ij_Q;
    ztensor<3>                _v_bar_ij_Q;

    bool                      _exch;
    // current leading index
    int                       _k0;
    long                      _current_chunk;
    long                      _chunk_size;
    long                      _NQ;
    const bz_utils_t&         _bz_utils;

    // base path to integral files
    std::string               _base_path;
  };

}  // namespace green::mbpt

#endif  // GF2_DFINTEGRAL_H
