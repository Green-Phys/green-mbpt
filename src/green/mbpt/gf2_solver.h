/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#ifndef MPIGF2_DFGF2SOLVER_H
#define MPIGF2_DFGF2SOLVER_H

#include <green/grids/transformer_t.h>
#include <green/ndarray/ndarray.h>
#include <green/ndarray/ndarray_math.h>
#include <green/params/params.h>
#include <green/symmetry/symmetry.h>
#include <green/utils/mpi_shared.h>
#include <green/utils/mpi_utils.h>
#include <green/utils/timing.h>
#include <mpi.h>

#include <Eigen/Core>

#include "common_defs.h"
#include "df_integral_t.h"

namespace green::mbpt {
  /**
   * @brief This class performs self-energy calculation by means of second-order PT using density fitting
   */
  class gf2_solver {
    using bz_utils_t = symmetry::brillouin_zone_utils<symmetry::inv_symm_op>;
    using G_type     = utils::shared_object<ztensor<5>>;
    using S1_type    = ztensor<4>;
    using St_type    = utils::shared_object<ztensor<5>>;

  public:
    /**
     * Class constructor
     * Initialize arrays and indices for GF2 loop
     *
     * @param nao -- number of atomic orbitals in cell
     * @param nts -- number of time steps
     * @param symm -- symmetrize selfenergy
     * @param nk -- number of k-points
     * @param NQ -- number of aux basis function in fitted densities
     * @param Gk -- Green's function array defined in (tau,ncell,nao,nao) domain
     * @param Sigma -- Self-energy array
     * @param path -- path to Integrals file
     */
    gf2_solver(const params::params& p, const grids::transformer_t& tr, const bz_utils_t& bz) :
        _nts(tr.sd().repn_fermi().nts()), _nk(bz.nk()), _ink(bz.ink()), _path(p["dfintegral_file"]),
        _ewald(p["dfintegral_file"].as<std::string>() != p["dfintegral_hf_file"].as<std::string>()), _bz_utils(bz),
        statistics("GF2") {
      h5pp::archive ar(p["input_file"]);
      ar["params/nao"] >> _nao;
      ar["params/nso"] >> _nso;
      ar["params/ns"] >> _ns;
      ar["params/NQ"] >> _NQ;
      ar.close();
    }

    /**
     * Solve GF2 equations for Self-energy
     */
    void solve(G_type& g_tau, S1_type& sigma1, St_type& sigma_tau);

  private:
    // dimension of problem (nao*ncell)
    size_t            _dim;
    // number of time steps
    size_t            _nts;

    size_t            _nk;
    size_t            _ink;
    size_t            _nao;
    size_t            _nso;
    size_t            _ns;
    size_t            _NQ;

    // Path to H5 file
    const std::string _path;

    // references to arrays
    ztensor<5>        Sigma_local;

    // Current time step Green's function matrix for k1
    Eigen::MatrixXcd  _G_k1_tmp;
    // Current reverse time step Green's function matrix for k2
    Eigen::MatrixXcd  _Gb_k2_tmp;
    // Current time step Green's function matrix for k3
    Eigen::MatrixXcd  _G_k3_tmp;

    /**
     * Read next part of Coulomb integrals for fixed set of k-points
     */
    void              read_next(const std::array<size_t, 4>& k);

    // Compute correction into second-order from the divergent G=0 part of the interaction
    // Compute second order contribution of the divergent part of the coulomb integrals
    void              compute_2nd_exch_correction(const ztensor<5>& Gr_full_tau);

    void      ewald_2nd_order_0_0(const ztensor<5>& Gr_full_tau, MatrixXcd& G1, MatrixXcd& G2, MatrixXcd& G3, MMatrixXcd& Xm_4,
                                  MMatrixXcd& Xm_1, MMatrixXcd& Xm_2, MMatrixXcd& Ym_1, MMatrixXcd& Ym_2, MMatrixXcd& Xm,
                                  MMatrixXcd& Vm);

    void      ewald_2nd_order_1_0(const ztensor<5>& Gr_full_tau, MatrixXcd& G1, MatrixXcd& G2, MatrixXcd& G3, MMatrixXcd& Xm_4,
                                  MMatrixXcd& Xm_1, MMatrixXcd& Xm_2, MMatrixXcd& Ym_1, MMatrixXcd& Ym_2, MMatrixXcd& Xm,
                                  MMatrixXcd& Vm);

    void      ewald_2nd_order_0_1(const ztensor<5>& Gr_full_tau, MatrixXcd& G1, MatrixXcd& G2, MatrixXcd& G3, MMatrixXcd& Xm_4,
                                  MMatrixXcd& Xm_1, MMatrixXcd& Xm_2, MMatrixXcd& Ym_1, MMatrixXcd& Ym_2, MMatrixXcd& Xm,
                                  MMatrixXcd& Vm);

    // Read next portion of the correction to a coulomb integral
    void      read_next_correction_0_0(size_t k);

    void      read_next_correction_1_0(size_t k1, size_t k2);

    void      read_next_correction_0_1(size_t k1, size_t k2);

    /**
     * Performs loop over time for fixed set of k-points
     */
    void      selfenergy_innerloop(size_t tau_offset, size_t ntau_local, const std::array<size_t, 4>& k, size_t is, const ztensor<5>& Gr_full_tau);

    MatrixXcd extract_G_tau_k(const ztensor<5>& G_tau, size_t t, size_t k_pos, size_t k_red, size_t s) {
      int         ts_shift = t * G_tau.shape()[1] * G_tau.shape()[2] * _nao * _nao + s * G_tau.shape()[2] * _nao * _nao;
      int         k_shift  = k_pos * _nao * _nao;
      CMMatrixXcd tmp(G_tau.data() + ts_shift + k_shift, G_tau.shape()[3], G_tau.shape()[4]);
      MatrixXcd   G = tmp;
      if (_bz_utils.symmetry().conj_list()[k_red] == 1) {
        for (size_t i = 0; i < _nao; ++i) {
          for (size_t j = 0; j < _nao; ++j) {
            G(i, j) = std::conj(G(i, j));
          }
        }
      }

      return G;
    }

    /**
     * Performs all possible contractions for i and n indices
     */
    void              contraction(size_t nao2, size_t nao3, bool eq_spin, bool ew_correct, const Eigen::MatrixXcd& G1,
                                  const Eigen::MatrixXcd& G2, const Eigen::MatrixXcd& G3, MMatrixXcd& Xm_4, MMatrixXcd& Xm_1, MMatrixXcd& Xm_2,
                                  MMatrixXcd& Ym_1, MMatrixXcd& Ym_2, const MMatrixXcd& vm_1, MMatrixXcd& Xm, MMatrixXcd& Vm, MMatrixXcd& Vxm,
                                  MMatrixXcd& Sm);

    /**
     * Compute two-electron integrals for the fixed set of k-points using pre-computed fitted densities
     *
     * @param set of k-points
     */
    void              setup_integrals(const std::array<size_t, 4>& k);

    // Pre-computed fitted densities
    // To avoid divergence in G=0 we separately compute ewald correction for the divergent part
    // Left interaction term
    df_integral_t*    _coul_int_c_1;
    df_integral_t*    _coul_int_c_2;
    // Right direct term
    df_integral_t*    _coul_int_c_3;
    df_integral_t*    _coul_int_c_4;
    // Right exchange term
    df_integral_t*    _coul_int_x_3;
    df_integral_t*    _coul_int_x_4;

    ztensor<4>        vijkl;
    ztensor<4>        vcijkl;
    ztensor<4>        vxijkl;
    ztensor<4>        vxcijkl;

    bool              _ewald;

    const bz_utils_t& _bz_utils;

    //
    utils::timing     statistics;
  };
}  // namespace green::mbpt

#endif  // MPIGF2_DFGF2SOLVER_H
