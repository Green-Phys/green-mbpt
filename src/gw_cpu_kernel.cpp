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

#include <green/mbpt/common_defs.h>
#include <green/mbpt/common_utils.h>
#include <green/integrals/df_integral_t.h>
#include <green/mbpt/kernels.h>
#include <green/utils/mpi_shared.h>
#include <green/utils/mpi_utils.h>

namespace green::mbpt::kernels {

  void gw_cpu_kernel::solve(G_type& g, St_type& sigma_tau) {
    int verbose_ints = (!utils::context.internode_rank) ? 1 : 0;
    _coul_int1 = new df_integral_t(_path, _nao, _NQ, _bz_utils, verbose_ints);
    utils::shared_object<ztensor<4>> P0_tilde_s(_nts, 1, _NQ, _NQ);
    utils::shared_object<ztensor<4>> Pw_tilde_s(_nw_b, 1, _NQ, _NQ);
    MPI_Datatype                     dt_matrix     = utils::create_matrix_datatype<std::complex<double>>(_nso * _nso);
    MPI_Op                           matrix_sum_op = utils::create_matrix_operation<std::complex<double>>();
    auto&                            sigma_fermi   = sigma_tau.object();
    _eps_inv_wq.set_zero();
    sigma_tau.fence();
    if (!utils::context.node_rank) sigma_fermi.set_zero();
    sigma_tau.fence();
    statistics.start("total");
    statistics.start("Main loop");
    for (size_t q = utils::context.internode_rank; q < _ink; q += utils::context.internode_size) {
      size_t q_ir = _bz_utils.symmetry().full_point(q);
      selfenergy_innerloop(q_ir, g, sigma_tau, P0_tilde_s, Pw_tilde_s);
    }
    statistics.end();
    statistics.start("selfenergy_reduce");
    sigma_tau.fence();
    if (!utils::context.node_rank) {
      utils::allreduce(MPI_IN_PLACE, sigma_fermi.data(), sigma_fermi.size() / (_nso * _nso), dt_matrix, matrix_sum_op,
                       utils::context.internode_comm);
      sigma_fermi /= _nk;
    }
    sigma_tau.fence();
    statistics.end();
    statistics.start("q0_correction");
    if (_q0_utils.q0_treatment() == extrapolate) {
      MPI_Allreduce(MPI_IN_PLACE, _eps_inv_wq.data(), _eps_inv_wq.size(), MPI_C_DOUBLE_COMPLEX, MPI_SUM, utils::context.global);
      _q0_utils.GW_q0_correction(_eps_inv_wq, sigma_fermi, g.object(), _ft, _X2C, utils::context.global_rank,
                                 utils::context.node_rank, utils::context.node_size, sigma_tau.win());
    }
    statistics.end();
    statistics.end();
    statistics.print(utils::context.global);
    MPI_Type_free(&dt_matrix);
    MPI_Op_free(&matrix_sum_op);
    delete _coul_int1;
  }

  void gw_cpu_kernel::selfenergy_innerloop(size_t q_ir, const G_type& G, St_type& Sigma,
                                           utils::shared_object<ztensor<4>>& P0_tilde_s,
                                           utils::shared_object<ztensor<4>>& Pw_tilde_s) {
#ifndef NDEBUG
    if (_nts % 2 != 0) {
      throw mbpt_wrong_grid("Number of tau points should be even!!!");
    }
#endif
    P0_tilde_s.fence();
    if (!utils::context.node_rank) P0_tilde_s.object().set_zero();
    P0_tilde_s.fence();

    auto [local_tau, tau_offset] = compute_local_and_offset_node_comm(_nts / 2);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, P0_tilde_s.win());
    for (size_t k1 = 0; k1 < _nk; ++k1) {
      std::array<size_t, 4> k = _bz_utils.momentum_conservation({
          {k1, 0, q_ir}
      });
      statistics.start("eval_P0_tilde");
      if (_p_sp) {  // Single-precision run
        eval_P0_tilde<std::complex<float>>(k, G, P0_tilde_s.object(), local_tau, tau_offset);
      } else {  // Double-precision run
        eval_P0_tilde<std::complex<double>>(k, G, P0_tilde_s.object(), local_tau, tau_offset);
      }
      statistics.end();
    }
    symmetrize_P0(P0_tilde_s.object(), local_tau, tau_offset);
    MPI_Win_sync(P0_tilde_s.win());
    MPI_Barrier(utils::context.node_comm);
    MPI_Win_unlock_all(P0_tilde_s.win());

    // statistics.start("P0_reduce");  // Reduction for different Q-Q'
    // if (!utils::context.internode_rank)
    //   utils::allreduce(MPI_IN_PLACE, P0_tilde_s.object().data(), P0_tilde_s.size(), MPI_C_DOUBLE_COMPLEX, MPI_SUM,
    //   utils::context.internode_comm);
    // statistics.end();
    statistics.start("eval_P_tilde");
    // Solve Dyson-like eqn of P(iOmega_{n})
    eval_P_tilde(q_ir, P0_tilde_s, Pw_tilde_s);
    statistics.end();
    MPI_Win_lock_all(MPI_MODE_NOCHECK, Sigma.win());
    for (size_t k1 = 0; k1 < _ink; ++k1) {
      size_t k1_ir = _bz_utils.symmetry().full_point(k1);
      // Loop over the degenerate points of q_ir
      for (size_t q_deg : _bz_utils.symmetry().deg(_bz_utils.symmetry().reduced_point(q_ir))) {
        std::array<size_t, 4> k = _bz_utils.momentum_conservation({
            {k1_ir, q_deg, 0}
        });
        statistics.start("eval_S");
        if (_sigma_sp) {
          eval_selfenergy<std::complex<float>>(k, G, Sigma, P0_tilde_s.object());
        } else {
          eval_selfenergy<std::complex<double>>(k, G, Sigma, P0_tilde_s.object());
        }
        statistics.end();
      }
    }
    MPI_Win_sync(Sigma.win());
    MPI_Barrier(utils::context.node_comm);
    MPI_Win_unlock_all(Sigma.win());
  }

  template <typename prec>
  void gw_cpu_kernel::eval_P0_tilde(const std::array<size_t, 4>& k, const G_type& G, ztensor<4>& P0_tilde, size_t local_tau,
                                    size_t tau_offset) {
    // k = (k1, 0, q_ir, k1+q_ir)
    // Link current k-points to the corresponding irreducible one
    // size_t k1 = _bz_utils.index()[k[0]];
    // size_t k1q = _bz_utils.index()[k[3]];

    // (Q, p, m) or (Q', t, n)*
    tensor<prec, 3> v(_NQ, _nao, _nao);
    _coul_int1->symmetrize(v, k[0], k[3]);
    MMatrixX<prec> vm(v.data(), _NQ, _nao * _nao);
    MMatrixX<prec> vmm(v.data(), _NQ * _nao, _nao);
    // #pragma omp parallel
    {
      MatrixX<prec>   Gb_k1(_nao, _nao);
      MatrixX<prec>   G_k1q(_nao, _nao);
      tensor<prec, 3> X1(_nao, _nao, _NQ);
      tensor<prec, 3> X2(_NQ, _nao, _nao);

      MMatrixX<prec>  VVm(X2.data(), _nao * _nao, _NQ);
      MMatrixX<prec>  VVmm(X2.data(), _nao, _nao * _NQ);
      MMatrixX<prec>  X1m(X1.data(), _nao, _nao * _NQ);
      MMatrixX<prec>  X2m(X2.data(), _NQ * _nao, _nao);
      MMatrixX<prec>  X1mm(X1.data(), _nao * _nao, _NQ);
      MMatrixX<prec>  X2mm(X2.data(), _NQ, _nao * _nao);

      double          prefactor = (_ns == 2 or _X2C) ? 1.0 : 2.0;
      // #pragma omp for
      size_t          pseudo_ns = (!_X2C) ? _ns : 4;
      size_t          a, b, i_shift, j_shift;
      for (size_t t = tau_offset, it = 0; it < local_tau; ++t, ++it) {  // Loop over half-tau
        size_t     tt = _nts - t - 1;                                   // beta - t
        MMatrixXcd P0(P0_tilde.data() + t * _NQ * _NQ, _NQ, _NQ);
        for (size_t s = 0; s < pseudo_ns; ++s) {
          if (!_X2C) {
            assign_G(k[0], tt, s, G.object(), Gb_k1);
            assign_G(k[3], t, s, G.object(), G_k1q);
          } else {
            a       = s / 2;
            b       = s % 2;
            i_shift = a * _nao;
            j_shift = b * _nao;

            assign_G_nso(k[0], tt, b, a, G.object(), Gb_k1);
            assign_G_nso(k[3], t, a, b, G.object(), G_k1q);
          }
          P0_contraction<prec>(Gb_k1, G_k1q, vm, VVm, VVmm, X1m, vmm, X2m, X1mm, X2mm, P0, prefactor);
        }
      }
    }
  }

  template <typename prec>
  void gw_cpu_kernel::assign_G(size_t k, size_t t, size_t s, const ztensor<5>& G_fermi, MatrixX<prec>& G_k) {
    // Symmetry related k
    // Find the position in the irreducible list
    size_t k_pos = _bz_utils.symmetry().reduced_point(k);
    for (size_t i = 0; i < _nao; ++i) {
      for (size_t j = 0; j < _nao; ++j) {
        G_k(i, j) =
            (prec)(_bz_utils.symmetry().conj_list()[k] == 0 ? G_fermi(t, s, k_pos, i, j) : std::conj(G_fermi(t, s, k_pos, i, j)));
      }
    }
  }

  template <typename prec>
  void gw_cpu_kernel::assign_G_nso(size_t k, size_t t, size_t s1, size_t s2, const ztensor<5>& G_fermi, MatrixX<prec>& G_k) {
    // Symmetry related k
    // Find the position in the irreducible list
    size_t k_pos   = _bz_utils.symmetry().reduced_point(k);

    size_t i_shift = s1 * _nao;
    size_t j_shift = s2 * _nao;
    for (size_t i = 0; i < _nao; ++i) {
      for (size_t j = 0; j < _nao; ++j) {
        if (_bz_utils.symmetry().conj_list()[k] == 0) {
          G_k(i, j) = (prec)(G_fermi(t, 0, k_pos, i + i_shift, j + j_shift));
        } else {
          // Inverse spin s1 and s2
          size_t ms1 = (s1 + 1) % 2;
          size_t ms2 = (s2 + 1) % 2;
          if (ms1 == ms2) {
            G_k(i, j) = (prec)(std::conj(G_fermi(t, 0, k_pos, i + ms1 * _nao, j + ms2 * _nao)));
          } else {
            G_k(i, j) = (prec)(-1.0 * std::conj(G_fermi(t, 0, k_pos, i + ms1 * _nao, j + ms2 * _nao)));
          }
        }
      }
    }
  }

  /**
   * Contraction of polarization function for given tau and k-point
   * @param t - [INPUT] imaginary time
   * @param k - [INPUT] [k1, k2, k1, k2]
   * @param q - [INPUT] k1 - k2
   */
  template <typename prec>
  void gw_cpu_kernel::P0_contraction(const MatrixX<prec>& Gb_k1, const MatrixX<prec>& G_k1q, MMatrixX<prec>& vm,
                                     MMatrixX<prec>& VVm, MMatrixX<prec>& VVmm, MMatrixX<prec>& X1m, MMatrixX<prec>& vmm,
                                     MMatrixX<prec>& X2m, MMatrixX<prec>& X1mm, MMatrixX<prec>& X2mm, MMatrixXcd& P0,
                                     double& prefactor) {
    statistics.start("P0_zgemm");
    // pm, Q
    VVm           = vm.transpose();
    // t,mQ = (t,p) * (p,mQ)
    X1m.noalias() = Gb_k1 * VVmm;
    // (Q't, m) = (Q't, n) * (n, m)
    X2m.noalias() = vmm.conjugate() * G_k1q.transpose();
    // double prefactor = (_ns==2)? 1.0 : 2.0;
    P0.transpose() -= prefactor * (X2mm * X1mm).template cast<std::complex<double>>();
    statistics.end();
  }

  void gw_cpu_kernel::symmetrize_P0(ztensor<4>& P0_tilde, size_t local_tau, size_t tau_offset) {
    for (size_t it = tau_offset, t = 0; t < local_tau; ++it, ++t) {  // Symmetry of tau: P0(t) = P0(beta - t)
      // Hermitization
      auto P0_t = P0_tilde(it);
      P0_t /= (_nk);
      make_hermitian(P0_t);
      matrix(P0_tilde(_nts - it - 1, 0)) = matrix(P0_tilde(it, 0));
    }
  }

  void gw_cpu_kernel::eval_P_tilde(int q_ir, utils::shared_object<ztensor<4>>& P0_tilde, utils::shared_object<ztensor<4>>& Pw_s) {
    // Transform P0_tilde from Fermionic tau to Bonsonic Matsubara grid
    auto [nw_local, w_offset] = compute_local_and_offset_node_comm(_nw_b);
    auto [nt_local, t_offset] = compute_local_and_offset_node_comm(_nts);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, Pw_s.win());
    auto & P0_w = Pw_s.object();
    statistics.start("P0(t) -> P0(w)");
    _ft.tau_f_to_w_b(P0_tilde.object(), P0_w, w_offset, nw_local, true);
    statistics.end();

    if (utils::context.global_rank == 0 && q_ir == 0) {
      print_leakage(_ft.check_chebyshev(P0_tilde.object()), "P0");
    }

    statistics.start("GW-BSE");
    // Solve Dyson-like eqn for ncheb frequency points
    MatrixXcd              identity = MatrixXcd::Identity(_NQ, _NQ);
    // Eigen::FullPivLU<MatrixXcd> lusolver(_NQ,_NQ);
    Eigen::LDLT<MatrixXcd> ldltsolver(_NQ);
    for (size_t n = w_offset, loc_n = 0; loc_n < nw_local; ++n, ++loc_n) {
      MatrixXcd temp     = identity - matrix(P0_w(n, 0));
      // temp = lusolver.compute(temp).inverse().eval();
      temp               = ldltsolver.compute(temp).solve(identity).eval();
      temp               = 0.5 * (temp + temp.conjugate().transpose().eval());
      matrix(P0_w(n, 0)) = (temp * matrix(P0_w(n, 0))).eval();
    }
    statistics.start("P reduce");
    MPI_Win_sync(Pw_s.win());
    MPI_Barrier(utils::context.node_comm);
    MPI_Win_unlock_all(Pw_s.win());
    statistics.end();
    statistics.end();

    statistics.start("P(w) -> P(t)");
    // Transform back from Bosonic Matsubara to Fermionic tau.
    //P0_tilde.fence();
    MPI_Win_lock_all(MPI_MODE_NOCHECK, P0_tilde.win());
    _ft.w_b_to_tau_f(P0_w, P0_tilde.object(), t_offset, nt_local, true);
    //P0_tilde.fence();
    // for G0W0 correction
    if (_q0_utils.q0_treatment() == extrapolate and utils::context.node_rank == 0) {
      size_t iq = _bz_utils.symmetry().reduced_point(q_ir);
      _q0_utils.aux_to_PW_00(P0_w, _eps_inv_wq, iq);
    }
    MPI_Win_sync(P0_tilde.win());
    MPI_Barrier(utils::context.node_comm);
    MPI_Win_unlock_all(P0_tilde.win());
    statistics.end();
    // Transform back to intermediate Chebyshev representation for P_tilde
    if (utils::context.global_rank == 0 && q_ir == 0) {
      print_leakage(_ft.check_chebyshev(P0_tilde.object()), "P");
    }
  }

  template <typename prec>
  void gw_cpu_kernel::eval_selfenergy(const std::array<size_t, 4>& k, const G_type& G_fermi, St_type& Sigma_fermi_s,
                                      ztensor<4>& P0_tilde) {
    // k = (k1_ir, q_deg, 0, k1_ir-q_deg)
    // Link to corresponding irreducible k-point
    size_t k1q_pos               = _bz_utils.symmetry().reduced_point(k[3]);
    auto [tau_local, tau_offset] = compute_local_and_offset_node_comm(_nts);
    auto&           Sigma_fermi  = Sigma_fermi_s.object();

    size_t          k1_pos       = _bz_utils.symmetry().reduced_point(k[0]);
    // (Q, i, m) or (Q', j, n)*
    tensor<prec, 3> v(_NQ, _nao, _nao);
    _coul_int1->symmetrize(v, k[0], k[3]);
    MMatrixX<prec> vm(v.data(), _NQ * _nao, _nao);

    // #pragma omp parallel
    {
      MatrixX<prec>   G_k1q(_nao, _nao);
      MatrixXcd       Sigma_ts(_nao, _nao);
      tensor<prec, 3> Y1(_NQ, _nao, _nao);
      tensor<prec, 3> Y2(_nao, _nao, _NQ);

      MMatrixX<prec>  Y1m(Y1.data(), _NQ * _nao, _nao);
      MMatrixX<prec>  Y1mm(Y1.data(), _NQ, _nao * _nao);
      MMatrixX<prec>  Y2mm(Y2.data(), _nao * _nao, _NQ);
      MMatrixX<prec>  X2m(Y1.data(), _nao, _NQ * _nao);
      MMatrixX<prec>  Y2mmm(Y2.data(), _nao, _nao * _NQ);
      MMatrixX<prec>  X2mm(Y1.data(), _nao * _NQ, _nao);

      // #pragma omp for
      size_t          pseudo_ns = (!_X2C) ? _ns : 4;
      size_t          a, b, i_shift, j_shift;
      size_t          sigma_shift;
      for (size_t t = tau_offset, it = 0; it < tau_local; ++t, ++it) {
        MMatrixXcd    P(P0_tilde.data() + t * _NQ * _NQ, _NQ, _NQ);
        MatrixX<prec> P_sp(_NQ, _NQ);
        P_sp = P.cast<prec>();
        for (size_t s = 0; s < pseudo_ns; ++s) {
          if (!_X2C) {
            assign_G(k[3], t, s, G_fermi.object(), G_k1q);
          } else {
            a       = s / 2;
            b       = s % 2;
            i_shift = a * _nao;
            j_shift = b * _nao;
            assign_G_nso(k[3], t, a, b, G_fermi.object(), G_k1q);
          }
          selfenergy_contraction(k, G_k1q, vm, Y1m, Y1mm, Y2mm, X2m, Y2mmm, X2mm, P_sp, Sigma_ts);
          // Write to shared memory self-energy
          if (!_X2C) {
            sigma_shift = t * _ns * _ink * _nao * _nao + s * _ink * _nao * _nao + k1_pos * _nao * _nao;
            MMatrixXcd Sm(Sigma_fermi.data() + sigma_shift, _nao, _nao);
            Sm.noalias() -= Sigma_ts;
          } else {
            sigma_shift = t * _ns * _ink * _nso * _nso + 0 * _ink * _nso * _nso + k1_pos * _nso * _nso;
            MMatrixXcd Sm_nso(Sigma_fermi.data() + sigma_shift, _nso, _nso);

            Sm_nso.block(a * _nao, b * _nao, _nao, _nao) -= Sigma_ts;
          }
        }
      }
    }
  }

  /**
   * Contraction for evaluating self-energy for given tau and k-point
   */
  template <typename prec>
  void gw_cpu_kernel::selfenergy_contraction(const std::array<size_t, 4>& k, const MatrixX<prec>& G_k1q, MMatrixX<prec>& vm,
                                             MMatrixX<prec>& Y1m, MMatrixX<prec>& Y1mm, MMatrixX<prec>& Y2mm, MMatrixX<prec>& X2m,
                                             MMatrixX<prec>& Y2mmm, MMatrixX<prec>& X2mm, MatrixX<prec>& P, MatrixXcd& Sm_ts) {
    statistics.start("Selfenergy_zgemm");
    // Qi,n = (Qi, m) * (m, n)
    Y1m.noalias() = vm * G_k1q;
    // (in, Q') = (in, Q) * (Q, Q')
    if (_bz_utils.symmetry().conj_list()[k[1]] == 0) {
      // q
      Y2mm.noalias() = Y1mm.transpose() * P;
    } else {
      // -q
      Y2mm.noalias() = Y1mm.transpose() * P.conjugate();
    }
    // n, Q'j
    X2m.noalias() = vm.transpose().conjugate();
    // ij = (i, nQ')*(nQ', j)
    Sm_ts         = (Y2mmm * X2mm).eval().template cast<std::complex<double>>();
    statistics.end();
  }

  // Explicit template instantiation
  template void gw_cpu_kernel::eval_P0_tilde<std::complex<float>>(const std::array<size_t, 4>& k, const G_type&, ztensor<4>&,
                                                                  size_t, size_t);
  template void gw_cpu_kernel::eval_P0_tilde<std::complex<double>>(const std::array<size_t, 4>& k, const G_type&, ztensor<4>&,
                                                                   size_t, size_t);
  template void gw_cpu_kernel::P0_contraction(const MatrixX<std::complex<float>>& Gb_k1,
                                              const MatrixX<std::complex<float>>& G_k1q, MMatrixX<std::complex<float>>& vm,
                                              MMatrixX<std::complex<float>>& VVm, MMatrixX<std::complex<float>>& VVmm,
                                              MMatrixX<std::complex<float>>& X1m, MMatrixX<std::complex<float>>& vmm,
                                              MMatrixX<std::complex<float>>& X2m, MMatrixX<std::complex<float>>& X1mm,
                                              MMatrixX<std::complex<float>>& X2mm, MMatrixXcd& P0, double& prefactor);
  template void gw_cpu_kernel::P0_contraction(const MatrixX<std::complex<double>>& Gb_k1,
                                              const MatrixX<std::complex<double>>& G_k1q, MMatrixX<std::complex<double>>& vm,
                                              MMatrixX<std::complex<double>>& VVm, MMatrixX<std::complex<double>>& VVmm,
                                              MMatrixX<std::complex<double>>& X1m, MMatrixX<std::complex<double>>& vmm,
                                              MMatrixX<std::complex<double>>& X2m, MMatrixX<std::complex<double>>& X1mm,
                                              MMatrixX<std::complex<double>>& X2mm, MMatrixXcd& P0, double& prefactor);
  template void gw_cpu_kernel::eval_selfenergy<std::complex<float>>(const std::array<size_t, 4>& k, const G_type&, St_type&,
                                                                    ztensor<4>&);
  template void gw_cpu_kernel::eval_selfenergy<std::complex<double>>(const std::array<size_t, 4>& k, const G_type&, St_type&,
                                                                     ztensor<4>&);
  template void gw_cpu_kernel::selfenergy_contraction(const std::array<size_t, 4>& k, const MatrixX<std::complex<float>>& G_k1q,
                                                      MMatrixX<std::complex<float>>& vm, MMatrixX<std::complex<float>>& Y1m,
                                                      MMatrixX<std::complex<float>>& Y1mm, MMatrixX<std::complex<float>>& Y2mm,
                                                      MMatrixX<std::complex<float>>& X2m, MMatrixX<std::complex<float>>& Y2mmm,
                                                      MMatrixX<std::complex<float>>& X2mm, MatrixX<std::complex<float>>& P,
                                                      MatrixXcd& Sm_ts);
  template void gw_cpu_kernel::selfenergy_contraction(const std::array<size_t, 4>& k, const MatrixX<std::complex<double>>& G_k1q,
                                                      MMatrixX<std::complex<double>>& vm, MMatrixX<std::complex<double>>& Y1m,
                                                      MMatrixX<std::complex<double>>& Y1mm, MMatrixX<std::complex<double>>& Y2mm,
                                                      MMatrixX<std::complex<double>>& X2m, MMatrixX<std::complex<double>>& Y2mmm,
                                                      MMatrixX<std::complex<double>>& X2mm, MatrixX<std::complex<double>>& P,
                                                      MatrixXcd& Sm_ts);
  template void gw_cpu_kernel::assign_G(size_t k, size_t t, size_t s, const ztensor<5>&, MatrixX<std::complex<double>>& G_k);
  template void gw_cpu_kernel::assign_G(size_t k, size_t t, size_t s, const ztensor<5>&, MatrixX<std::complex<float>>& G_k);
  template void gw_cpu_kernel::assign_G_nso(size_t k, size_t t, size_t s1, size_t s2, const ztensor<5>&,
                                            MatrixX<std::complex<double>>& G_k);
  template void gw_cpu_kernel::assign_G_nso(size_t k, size_t t, size_t s1, size_t s2, const ztensor<5>&,
                                            MatrixX<std::complex<float>>& G_k);

}
