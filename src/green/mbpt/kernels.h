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

  class gw_kernel {
  public:
    using bz_utils_t = symmetry::brillouin_zone_utils<symmetry::inv_symm_op>;
    using G_type     = utils::shared_object<ztensor<5>>;
    using St_type    = utils::shared_object<ztensor<5>>;

    gw_kernel(const params::params& p, size_t nao, size_t nso, size_t ns, size_t NQ, bool X2C, const grids::transformer_t& ft,
              const bz_utils_t& bz_utils, const ztensor<4>& S_k) :
        _beta(p["BETA"]), _nts(ft.sd().repn_fermi().nts()), _nts_b(ft.sd().repn_bose().nts()), _ni(ft.sd().repn_fermi().ni()),
        _ni_b(ft.sd().repn_bose().ni()), _nw(ft.sd().repn_fermi().nw()), _nw_b(ft.sd().repn_bose().nw()), _nk(bz_utils.nk()),
        _ink(bz_utils.ink()), _nao(nao), _nso(nso), _ns(ns), _NQ(NQ), _X2C(X2C), _p_sp(p["P_sp"]), _sigma_sp(p["Sigma_sp"]),
        _ft(ft), _bz_utils(bz_utils), _path(p["dfintegral_file"]), _q0_utils(bz_utils.ink(), 0, S_k, _path, p["q0_treatment"]),
        _P0_tilde(0, 0, 0, 0), _eps_inv_wq(ft.wsample_bose().size(), bz_utils.ink()), _ntauspin_mpi(p["ntauspinprocs"]) {
      _q0_utils.resize(_NQ);
    }

    virtual ~    gw_kernel() = default;

    virtual void solve(G_type& g, St_type& sigma_tau);

  private:
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

    /**
     * Setup subcommunicator, _tau_comm2, over tau axis for a given k-point q.
     * @param q - [INPUT] k-point index
     */
    void setup_subcommunicator2(int q);

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
    //
    mbpt_q0_utils_t             _q0_utils;
    // Array for the polarization bubble and for screened interaction
    ztensor<4>                  _P0_tilde;
    // Dielectric function inverse in the plane-wave basis with G = G' = 0
    ztensor<2>                  _eps_inv_wq;

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

    utils::timing               statistics;
  };

  class gw_scalar_cpu_kernel : public gw_kernel {
  public:
    gw_scalar_cpu_kernel(const params::params& p, size_t nao, size_t nso, size_t ns, size_t NQ, const grids::transformer_t& ft,
                         const bz_utils_t& bz_utils, const ztensor<4>& S_k) :
        gw_kernel(p, nao, nso, ns, NQ, false, ft, bz_utils, S_k) {}
  };
  class gw_x2c_cpu_kernel : public gw_kernel {
  public:
    gw_x2c_cpu_kernel(const params::params& p, size_t nao, size_t nso, size_t ns, size_t NQ, const grids::transformer_t& ft,
                      const bz_utils_t& bz_utils, const ztensor<4>& S_k) :
        gw_kernel(p, nao, nso, ns, NQ, true, ft, bz_utils, S_k) {}
  };

  class hf_kernel {
  public:
    using bz_utils_t = symmetry::brillouin_zone_utils<symmetry::inv_symm_op>;
    using dm_type    = ztensor<4>;
    using S1_type    = ztensor<4>;

    hf_kernel(const params::params& p, size_t nao, size_t nso, size_t ns, size_t NQ, double madelung, const bz_utils_t& bz_utils,
              const ztensor<4>& S_k) :
        _nao(nao), _nso(nso), _nk(bz_utils.nk()), _ink(bz_utils.ink()), _ns(ns), _NQ(NQ), _madelung(madelung),
        _bz_utils(bz_utils), _S_k(S_k), _hf_path(p["dfintegral_hf_file"]) {}

    virtual S1_type solve(const dm_type& dm) = 0;
    virtual ~       hf_kernel()              = default;

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
    S1_type solve(const dm_type& dm) override;
  };

  class hf_x2c_cpu_kernel : public hf_kernel {
  public:
    hf_x2c_cpu_kernel(const params::params& p, size_t nao, size_t nso, size_t ns, size_t NQ, double madelung,
                      const bz_utils_t& bz_utils, const ztensor<4>& S_k) :
        hf_kernel(p, nao, nso, ns, NQ, madelung, bz_utils, S_k) {}
    S1_type solve(const dm_type& dm) override;

  private:
    MatrixXcd compute_exchange(int ik, ztensor<3>& dm_s1_s2, ztensor<3>& dm_ms1_ms2, ztensor<3>& v, df_integral_t& coul_int1,
                               ztensor<3>& Y, MMatrixXcd& Ym, MMatrixXcd& Ymm, ztensor<3>& Y1, MMatrixXcd& Y1m, MMatrixXcd& Y1mm,
                               MMatrixXcd& vmm, ztensor<3>& v2, MMatrixXcd& v2m, MMatrixXcd& v2mm);
    MatrixXcd compute_exchange_ab(int ik, ztensor<3>& dm_ab, ztensor<3>& v, df_integral_t& coul_int1, ztensor<3>& Y,
                                  MMatrixXcd& Ym, MMatrixXcd& Ymm, ztensor<3>& Y1, MMatrixXcd& Y1m, MMatrixXcd& Y1mm,
                                  MMatrixXcd& vmm, ztensor<3>& v2, MMatrixXcd& v2m, MMatrixXcd& v2mm);
  };

  class hf_kernel_factory {
    using bz_utils_t = symmetry::brillouin_zone_utils<symmetry::inv_symm_op>;

  public:
    static std::unique_ptr<hf_kernel> get_kernel(bool X2C, const params::params& p, size_t nao, size_t nso, size_t ns, size_t NQ,
                                                 double madelung, const bz_utils_t& bz_utils, const ztensor<4>& S_k) {
      if (X2C) {
        if (p["kernel"].as<kernel_type>() == CPU) {
          return std::unique_ptr<hf_x2c_cpu_kernel>(new hf_x2c_cpu_kernel(p, nao, nso, ns, NQ, madelung, bz_utils, S_k));
        }
      } else {
        if (p["kernel"].as<kernel_type>() == CPU) {
          return std::unique_ptr<hf_scalar_cpu_kernel>(new hf_scalar_cpu_kernel(p, nao, nso, ns, NQ, madelung, bz_utils, S_k));
        }
      }
      throw mbpt_kernel_error("Cannot determine HF kernel");
    }
  };

  class gw_kernel_factory {
    using bz_utils_t = symmetry::brillouin_zone_utils<symmetry::inv_symm_op>;

  public:
    static std::unique_ptr<gw_kernel> get_kernel(bool X2C, const params::params& p, size_t nao, size_t nso, size_t ns, size_t NQ,
                                                 const grids::transformer_t& ft, const bz_utils_t& bz_utils,
                                                 const ztensor<4>& S_k) {
      if (X2C) {
        if (p["kernel"].as<kernel_type>() == CPU) {
          return std::unique_ptr<gw_x2c_cpu_kernel>(new gw_x2c_cpu_kernel(p, nao, nso, ns, NQ, ft, bz_utils, S_k));
        }
      } else {
        if (p["kernel"].as<kernel_type>() == CPU) {
          return std::unique_ptr<gw_scalar_cpu_kernel>(new gw_scalar_cpu_kernel(p, nao, nso, ns, NQ, ft, bz_utils, S_k));
        }
      }
      throw mbpt_kernel_error("Cannot determine GW kernel");
    }
  };
}  // namespace green::mbpt::kernels

#endif  // GREEN_MBPT_GW_KERNEL_H
