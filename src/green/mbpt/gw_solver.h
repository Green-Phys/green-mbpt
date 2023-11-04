/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#ifndef GREEN_GW_SOLVER_T_H
#define GREEN_GW_SOLVER_T_H

#include <green/grids/transformer_t.h>
#include <green/ndarray/ndarray.h>
#include <green/ndarray/ndarray_math.h>
#include <green/params/params.h>
#include <green/symmetry/symmetry.h>
#include <green/utils/mpi_shared.h>
#include <green/utils/mpi_utils.h>
#include <green/utils/timing.h>
#include <mpi.h>

// #include "gscf/gscf_cuhf_solver_t.h"
// #include "transformer_t.h"
#include "common_defs.h"
#include "df_integral_t.h"

namespace green::mbpt {
  /**
   * @brief GWSolver class performs self-energy calculation by means of GW approximation using density fitting
   */
  class gw_solver {
    using bz_utils_t = symmetry::brillouin_zone_utils<symmetry::inv_symm_op>;
    using G_type     = utils::shared_object<ztensor<5>>;
    using S1_type    = ztensor<4>;
    using St_type    = utils::shared_object<ztensor<5>>;

  public:
    /**
     * Class constructor
     *
     * @param p           -- simulation parameters
     * @param ft          -- imaginary time transformer
     * @param Gk          -- Green's function in (tau, kcell, nao, nao) domain
     * @param Sigma       -- Self-energy in (tau, kcell, nao, nao) domain
     * @param bz_utils    -- Brillouin zone utilities
     * @param second_only -- Whether do GW or only second-order direct diagram
     */
    gw_solver(const params::params& p, const grids::transformer_t& ft, const bz_utils_t& bz_utils, bool second_only = false) :
        _beta(p["BETA"]), _nts(ft.sd().repn_fermi().nts()), _nts_b(ft.sd().repn_bose().nts()), _ni(ft.sd().repn_fermi().ni()),
        _ni_b(ft.sd().repn_bose().ni()), _nw(ft.sd().repn_fermi().nw()), _nw_b(ft.sd().repn_bose().nw()), _nk(bz_utils.nk()),
        _ink(bz_utils.ink()), _second_only(second_only), _ft(ft), _bz_utils(bz_utils), _path(p["dfintegral_file"]),
        _P0_tilde(0, 0, 0, 0), _ntauspin_mpi(p["ntauspinprocs"]), _p_sp(p["P_sp"]), _sigma_sp(p["Sigma_sp"]) {
      h5pp::archive ar(p["input_file"]);
      ar["params/nao"] >> _nao;
      ar["params/nso"] >> _nso;
      ar["params/ns"] >> _ns;
      ar["params/NQ"] >> _NQ;
      ar.close();
      _X2C = _nao != _nso;
      _P0_tilde.resize(_nts, 1, _NQ, _NQ);
    }

    /**
     * Solve GW equations for Self-energy
     */
    void solve(G_type& g, S1_type& sigma1, St_type& sigma_tau);

    /**
     * Evaluate self-energy contribution from P^{q_ir}
     * @param q_ir - [INPUT] momentum index of polarization and screened interaction
     * @param subcomm - [INPUT] Sub-communicator for tau and spin axis.
     */
    void selfenergy_innerloop(size_t q_ir, MPI_Comm subcomm, const G_type& G_fermi, St_type& Sigma_fermi_s);

    /**
     * Divide all (_ink) MPI jobs into batches, only kbatch_size of k-points is processed at a time.
     * For each k-point, we setup subcommunicator, _tauspin_comm1, to parallel over tau and spin axes.
     * @param kbatch_size - [OUTPUT] number of k-points to be processed at a time = _nprocs / _ntauspinprocs
     * @param num_kbatch - [OUTPUT] number of k-batches = _ink / kbatch_size
     */
    void setup_subcommunicator(size_t& kbatch_size, size_t& num_kbatch);

  protected:
    /**
     * Setup subcommunicator, _tau_comm2, over tau axis for a given k-point q.
     * @param q - [INPUT] k-point index
     */
    void setup_subcommunicator2(int q);

    template <size_t N>
    void hermitization(ztensor<N>& X) {
      // Dimension of the rest of arrays
      size_t dim1 = std::accumulate(X.shape().begin(), X.shape().end() - 2, 1ul, std::multiplies<size_t>());
      size_t nao  = X.shape()[N - 1];
      for (size_t i = 0; i < dim1; ++i) {
        MMatrixXcd Xm(X.data() + i * nao * nao, nao, nao);
        Xm = 0.5 * (Xm + Xm.conjugate().transpose().eval());
      }
    }

    /**
     * Read next part of Coulomb integrals in terms of 3-index tensors for fixed set of k-points
     * @param k - [INPUT] (k1, 0, q, k1+q) or (k1, q, 0, k1-q)
     */
    void read_next(const std::array<size_t, 4>& k);

    /**
     * Evaluate polarization function P for a given job portion (maybe a single k-point or a set of k-points),
     * in desired precision
     */
    template <typename prec>
    void eval_P0_tilde(const std::array<size_t, 4>& k, const G_type& G);

    template <typename prec>
    void assign_G(size_t k, size_t t, size_t s, const ztensor<5>& G_fermi, MatrixX<prec>& G_k);
    template <typename prec>
    void assign_G_nso(size_t k, size_t t, size_t s1, size_t s2, const ztensor<5>& G_fermi, MatrixX<prec>& G_k);

    /**
     * Contraction of polarization function for given tau and k-point
     * @param t - [INPUT] imaginary time
     * @param k - [INPUT] [k1, k2, k1, k2]
     * @param q - [INPUT] k1 - k2
     */
    template <typename prec>
    void P0_contraction(const MatrixX<prec>& Gb_k1, const MatrixX<prec>& G_k1q, MMatrixX<prec>& vm, MMatrixX<prec>& VVm,
                        MMatrixX<prec>& VVmm, MMatrixX<prec>& X1m, MMatrixX<prec>& vmm, MMatrixX<prec>& X2m, MMatrixX<prec>& X1mm,
                        MMatrixX<prec>& X2mm, MMatrixXcd& P0, double& prefactor);

    /**
     * Symmetrize polarization function by ultilizing P0(t) = P0(beta-t)
     */
    void symmetrize_P0();

    /**
     * Solve Dyson-like equation for screened interaction W using Chebyshev convolution
     * Writes the resulting P_tilde(tau) in_P0_tilde;
     */
    void eval_P_tilde(int q_ir);

    /**
     * Takes P0_tilde(tau) and
     * solves Dyson-like equation for screened interaction W using Chebyshev convolution
     * for a specific
     * Writes the resulting P_tilde(Omega) in the argument P_w.
     * This function is needed separately for two-particle density matrix evaluation.
     */
    void eval_P_tilde_w(int q_ir, ztensor<4>& P0_tilde, ztensor<4>& P_w);

    /**
     * Evaluate self-energy
     */
    template <typename prec>
    void eval_selfenergy(const std::array<size_t, 4>& k, const G_type& G_fermi, St_type& Sigma_fermi_s);

    /**
     * Contraction for evaluating self-energy for given tau and k-point
     */
    template <typename prec>
    void selfenergy_contraction(const std::array<size_t, 4>& k, const MatrixX<prec>& G_k1q, MMatrixX<prec>& vm,
                                MMatrixX<prec>& Y1m, MMatrixX<prec>& Y1mm, MMatrixX<prec>& Y2mm, MMatrixX<prec>& X2m,
                                MMatrixX<prec>& Y2mmm, MMatrixX<prec>& X2mm, MatrixX<prec>& P, MatrixXcd& Sm_ts);

  protected:
    double                      _beta;
    size_t                      _nts;
    size_t                      _nts_b;
    size_t                      _ni;
    size_t                      _ni_b;
    size_t                      _nw;
    size_t                      _nw_b;

    size_t                      _nk;
    size_t                      _ink;

    bool                        _p_sp;
    bool                        _sigma_sp;
    // Compute the 2nd-order direct diagram only
    bool                        _second_only;

    const grids::transformer_t& _ft;
    const bz_utils_t&           _bz_utils;
    // Path to integral files
    const std::string           _path;
    // Array for the polarization bubble and for screened interaction
    ztensor<4>                  _P0_tilde;

    size_t                      _nao;
    size_t                      _nso;
    size_t                      _ns;
    size_t                      _NQ;
    bool                        _X2C;
    // MPI communicators to be used
    MPI_Comm                    _tauspin_comm;
    MPI_Comm                    _tau_comm2;
    // Number of total processors and MPI jobs on tau+spin axes. Both are specified by users.
    int                         _ntauspin_mpi;
    // Processors' information
    int                         _ntauprocs;
    int                         _nspinprocs;
    int                         _tauid;
    int                         _spinid;

    // Pre-computed fitted densities
    // This object reads 3-index tensors into Vij_Q
    df_integral_t*              _coul_int1;
    df_integral_t*              _coul_int2;

    utils::timing               statistics;
  };

}  // namespace green::mbpt

#endif  // GF2_GW_SOLVER_T_H
