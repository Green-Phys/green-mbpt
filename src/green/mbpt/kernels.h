/*
 * Copyright (c) 2023 University of Michigan
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the “Software”), to deal in the Software
 * without restriction, including without limitation the rights to use, copy, modify,
 * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef GREEN_MBPT_GW_KERNEL_H
#define GREEN_MBPT_GW_KERNEL_H
#include <green/grids/transformer_t.h>
#include <green/symmetry/symmetry.h>
#include <green/utils/mpi_shared.h>
#include <green/utils/mpi_utils.h>
#include <green/utils/timing.h>

#include "common_defs.h"
#include "df_integral_t.h"
#include "except.h"
#include "mbpt_q0_utils_t.h"

namespace green::mbpt::kernels {
  class gw_cpu_kernel {
    using bz_utils_t = symmetry::brillouin_zone_utils<symmetry::inv_symm_op>;
    using G_type     = utils::shared_object<ztensor<5>>;
    using St_type    = utils::shared_object<ztensor<5>>;

  public:
    gw_cpu_kernel(const params::params& p, size_t nao, size_t nso, size_t ns, size_t NQ, const grids::transformer_t& ft,
                  const bz_utils_t& bz_utils, const ztensor<4>& S_k, bool X2C = false) :
        _beta(p["BETA"]), _nts(ft.sd().repn_fermi().nts()), _nts_b(ft.sd().repn_bose().nts()), _ni(ft.sd().repn_fermi().ni()),
        _ni_b(ft.sd().repn_bose().ni()), _nw(ft.sd().repn_fermi().nw()), _nw_b(ft.sd().repn_bose().nw()), _nk(bz_utils.nk()),
        _ink(bz_utils.ink()), _nao(nao), _nso(nso), _ns(ns), _NQ(NQ), _X2C(X2C), _p_sp(p["P_sp"]), _sigma_sp(p["Sigma_sp"]),
        _ft(ft), _bz_utils(bz_utils), _path(p["dfintegral_file"]), statistics("GW"),
        _q0_utils(bz_utils.ink(), 0, S_k, _path, p["q0_treatment"]),
        // _P0_tilde(0, 0, 0, 0),
        _eps_inv_wq(ft.wsample_bose().size(), bz_utils.ink()),
        _coul_int1(nullptr) {
      _q0_utils.resize(_NQ);
    }

    void solve(G_type& g, St_type& sigma_tau);

  private:
    double                      _beta;
    size_t                      _nts;
    size_t                      _nts_b;
    size_t                      _ni;
    size_t                      _ni_b;
    size_t                      _nw;
    size_t                      _nw_b;

    size_t                      _nk;
    size_t                      _ink;
    size_t                      _nao;
    size_t                      _nso;
    size_t                      _ns;
    size_t                      _NQ;
    bool                        _X2C;

    bool                        _p_sp;
    bool                        _sigma_sp;

    const grids::transformer_t& _ft;
    const bz_utils_t&           _bz_utils;
    // Path to integral files
    const std::string           _path;
    utils::timing               statistics;
    //
    mbpt_q0_utils_t             _q0_utils;
    // Array for the polarization bubble and for screened interaction
    // ztensor<4>                  _P0_tilde;
    // Dielectric function inverse in the plane-wave basis with G = G' = 0
    ztensor<2>                  _eps_inv_wq;

    // // MPI communicators to be used
    // MPI_Comm                    _tauspin_comm;
    // MPI_Comm                    _tau_comm2;
    // // Number of total processors and MPI jobs on tau+spin axes. Both are specified by users.
    // int                         _ntauspin_mpi;
    // // Processors' information
    // int                         _ntauprocs;
    // int                         _nspinprocs;
    // int                         _tauid;
    // int                         _spinid;
    // Pre-computed fitted densities
    // This object reads 3-index tensors into Vij_Q
    df_integral_t*              _coul_int1;

  private:
    /**
     * Evaluate self-energy contribution from P^{q_ir}
     * @param q_ir - [INPUT] momentum index of polarization and screened interaction
     */
    void selfenergy_innerloop(size_t q_ir, const G_type& G_fermi, St_type& Sigma_fermi_s,
                              utils::shared_object<ztensor<4>>& P0_tilde_s, utils::shared_object<ztensor<4>>& Pw_tilde_s);

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
    void eval_P0_tilde(const std::array<size_t, 4>& k, const G_type& G, ztensor<4>& P0_tilde_s, size_t local_tau,
                       size_t tau_offset);

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
    void symmetrize_P0(ztensor<4>& P0_tilde, size_t local_tau, size_t tau_offset);

    /**
     * Solve Dyson-like equation for screened interaction W using Chebyshev convolution
     * Writes the resulting P_tilde(tau) in_P0_tilde;
     */
    void eval_P_tilde(int q_ir, utils::shared_object<ztensor<4>>& P0_tilde_s, utils::shared_object<ztensor<4>>& Pw_s);

    /**
     * Evaluate self-energy
     */
    template <typename prec>
    void eval_selfenergy(const std::array<size_t, 4>& k, const G_type& G_fermi, St_type& Sigma_fermi_s, ztensor<4>& P0_tilde);

    /**
     * Contraction for evaluating self-energy for given tau and k-point
     */
    template <typename prec>
    void selfenergy_contraction(const std::array<size_t, 4>& k, const MatrixX<prec>& G_k1q, MMatrixX<prec>& vm,
                                MMatrixX<prec>& Y1m, MMatrixX<prec>& Y1mm, MMatrixX<prec>& Y2mm, MMatrixX<prec>& X2m,
                                MMatrixX<prec>& Y2mmm, MMatrixX<prec>& X2mm, MatrixX<prec>& P, MatrixXcd& Sm_ts);
  };

  class hf_kernel {
  public:
    using bz_utils_t = symmetry::brillouin_zone_utils<symmetry::inv_symm_op>;
    using dm_type    = ztensor<4>;
    using S1_type    = ztensor<4>;

    hf_kernel(const params::params& p, size_t nao, size_t nso, size_t ns, size_t NQ, double madelung, const bz_utils_t& bz_utils,
              const ztensor<4>& S_k) :
        _nao(nao), _nso(nso), _nk(bz_utils.nk()), _ink(bz_utils.ink()), _ns(ns), _NQ(NQ), _madelung(madelung),
        _bz_utils(bz_utils), _S_k(S_k), _hf_path(p["dfintegral_hf_file"]), statistics("Hartree Fock"){};
    virtual ~hf_kernel() = default;

  protected:
    // number of atomic orbitals per cell
    size_t            _nao;
    size_t            _nso;
    // number of cells for GF2 loop
    size_t            _nk;
    // number of k-point after time-reversal symmetry
    size_t            _ink;
    // number of spins
    size_t            _ns;
    // auxiliraly basis size
    size_t            _NQ;
    // madelung constant
    double            _madelung;
    const bz_utils_t& _bz_utils;
    // overlap
    const ztensor<4>& _S_k;
    const std::string _hf_path;
    utils::timing     statistics;
  };

  class hf_scalar_cpu_kernel final : public hf_kernel {
  public:
    hf_scalar_cpu_kernel(const params::params& p, size_t nao, size_t nso, size_t ns, size_t NQ, double madelung,
                         const bz_utils_t& bz_utils, const ztensor<4>& S_k) :
        hf_kernel(p, nao, nso, ns, NQ, madelung, bz_utils, S_k) {}
    S1_type solve(const dm_type& dm);
  };

  class hf_x2c_cpu_kernel : public hf_kernel {
  public:
    hf_x2c_cpu_kernel(const params::params& p, size_t nao, size_t nso, size_t ns, size_t NQ, double madelung,
                      const bz_utils_t& bz_utils, const ztensor<4>& S_k) :
        hf_kernel(p, nao, nso, ns, NQ, madelung, bz_utils, S_k) {}
    S1_type solve(const dm_type& dm);

  private:
    MatrixXcd compute_exchange(int ik, ztensor<3>& dm_s1_s2, ztensor<3>& dm_ms1_ms2, ztensor<3>& v, df_integral_t& coul_int1,
                               ztensor<3>& Y, MMatrixXcd& Ym, MMatrixXcd& Ymm, ztensor<3>& Y1, MMatrixXcd& Y1m, MMatrixXcd& Y1mm,
                               MMatrixXcd& vmm, ztensor<3>& v2, MMatrixXcd& v2m, MMatrixXcd& v2mm);
    MatrixXcd compute_exchange_ab(int ik, ztensor<3>& dm_ab, ztensor<3>& v, df_integral_t& coul_int1, ztensor<3>& Y,
                                  MMatrixXcd& Ym, MMatrixXcd& Ymm, ztensor<3>& Y1, MMatrixXcd& Y1m, MMatrixXcd& Y1mm,
                                  MMatrixXcd& vmm, ztensor<3>& v2, MMatrixXcd& v2m, MMatrixXcd& v2mm);
  };
}  // namespace green::mbpt::kernels

#endif  // GREEN_MBPT_GW_KERNEL_H
