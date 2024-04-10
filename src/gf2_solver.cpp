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

#include "green/mbpt/gf2_solver.h"

#include <green/mbpt/common_utils.h>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Sparse>
#include <fstream>

namespace green::mbpt {
  void gf2_solver::solve(G_type& g_tau, S1_type& sigma1, St_type& sigma_tau) {
    _G_k1_tmp.resize(_nao, _nao);
    _Gb_k2_tmp.resize(_nao, _nao);
    _G_k3_tmp.resize(_nao, _nao);
    vijkl.resize(_nao, _nao, _nao, _nao);
    vcijkl.resize(_nao, _nao, _nao, _nao);
    vxijkl.resize(_nao, _nao, _nao, _nao);
    vxcijkl.resize(_nao, _nao, _nao, _nao);
    //
    MPI_Datatype dt_matrix     = utils::create_matrix_datatype<std::complex<double>>(_nso * _nso);
    MPI_Op       matrix_sum_op = utils::create_matrix_operation<std::complex<double>>();
    _coul_int_c_1              = new df_integral_t(_path, _nao, _NQ, _bz_utils);
    _coul_int_c_2              = new df_integral_t(_path, _nao, _NQ, _bz_utils);
    _coul_int_c_3              = new df_integral_t(_path, _nao, _NQ, _bz_utils);
    _coul_int_c_4              = new df_integral_t(_path, _nao, _NQ, _bz_utils);
    _coul_int_x_3              = new df_integral_t(_path, _nao, _NQ, _bz_utils);
    _coul_int_x_4              = new df_integral_t(_path, _nao, _NQ, _bz_utils);
    auto& Sigma_tau            = sigma_tau.object();
    // clean self_energy array
    Sigma_local                = Sigma_tau;
    sigma_tau.fence();
    if (!utils::context.node_rank) Sigma_tau.set_zero();
    sigma_tau.fence();
    statistics.start("GF2 total");
    auto [ntau_local, tau_offset] = compute_local_and_offset_node_comm(_nts);
    // start main loop
    MPI_Win_lock_all(MPI_MODE_NOCHECK, sigma_tau.win());
    for (size_t k1k3k2 = utils::context.internode_rank; k1k3k2 < _nk * _nk * _ink; k1k3k2 += utils::context.internode_size) {
      size_t                k1_pos = k1k3k2 / (_nk * _nk);
      // Link the reduce index (k1_pos) to corresponding momentum
      size_t                k1_red = _bz_utils.symmetry().full_point(k1_pos);
      size_t                k3     = (k1k3k2 / _nk) % _nk;
      size_t                k2     = k1k3k2 % _nk;
      std::array<size_t, 4> k      = _bz_utils.momentum_conservation({k1_red, k2, k3});
      statistics.start("read");
      // read next part of integrals
      read_next(k);
      statistics.end();
      statistics.start("setup");
      setup_integrals(k);
      statistics.end();
      for (size_t is = 0; is < _ns; ++is) {
        selfenergy_innerloop(tau_offset, ntau_local, k, is, g_tau.object());
      }
    }
    MPI_Win_sync(sigma_tau.win());
    MPI_Barrier(utils::context.node_comm);
    MPI_Win_unlock_all(sigma_tau.win());
    MPI_Barrier(utils::context.global);
    statistics.start("correction");
    if (_ewald) {
      if (!utils::context.global_rank) std::cout << "Correction for selfenergy" << std::endl;
      compute_2nd_exch_correction(g_tau.object());
    }
    statistics.end();

    statistics.start("reduce");
    // // collect data within a node
    // MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, sigma_tau.win());
    // Sigma_tau += Sigma_local;
    // MPI_Win_unlock(0, sigma_tau.win());
    // collect data among nodes
    sigma_tau.fence();
    if (!utils::context.node_rank) {
      utils::allreduce(MPI_IN_PLACE, Sigma_tau.data(), Sigma_tau.size() / (_nso * _nso), dt_matrix, matrix_sum_op,
                       utils::context.internode_comm);
      Sigma_tau /= (_nk * _nk);
    }
    sigma_tau.fence();
    statistics.end();
    statistics.end();
    // print execution time
    statistics.print(utils::context.global);
    delete _coul_int_c_1;
    delete _coul_int_c_2;
    delete _coul_int_c_3;
    delete _coul_int_c_4;
    delete _coul_int_x_3;
    delete _coul_int_x_4;
    _G_k1_tmp.resize(0, 0);
    _Gb_k2_tmp.resize(0, 0);
    _G_k3_tmp.resize(0, 0);
    vijkl.resize(0, 0, 0, 0);
    vcijkl.resize(0, 0, 0, 0);
    vxijkl.resize(0, 0, 0, 0);
    vxcijkl.resize(0, 0, 0, 0);
    MPI_Type_free(&dt_matrix);
    MPI_Op_free(&matrix_sum_op);
  }

  void gf2_solver::selfenergy_innerloop(size_t tau_offset, size_t ntau_local, const std::array<size_t, 4>& k, size_t is, const ztensor<5>& Gr_full_tau) {
    statistics.start("nao");
    size_t nao2     = _nao * _nao;
    size_t nao3     = _nao * _nao * _nao;
    // Link current k-points to corresponding reduced k's
    // k = (k_red, k1, k2, k3)
    // Find the position in the irreducible list
    size_t k1_pos   = _bz_utils.symmetry().reduced_point(k[1]);
    size_t k2_pos   = _bz_utils.symmetry().reduced_point(k[2]);
    size_t k3_pos   = _bz_utils.symmetry().reduced_point(k[3]);
    size_t k0_pos   = _bz_utils.symmetry().reduced_point(k[0]);
    size_t momshift = k0_pos * nao2;
#pragma omp parallel
    {
      Eigen::MatrixXcd G1(_nao, _nao);
      Eigen::MatrixXcd G2(_nao, _nao);
      Eigen::MatrixXcd G3(_nao, _nao);
      ztensor<3>       X(_nao, _nao, _nao);
      MMatrixXcd       Xm_4(X.data(), _nao, nao2);
      ztensor<3>       X1(_nao, _nao, _nao);
      MMatrixXcd       Xm_1(X1.data(), nao2, _nao);
      MMatrixXcd       Xm_2(X1.data(), _nao, nao2);
      ztensor<3>       Y1(_nao, _nao, _nao);
      MMatrixXcd       Ym_1(Y1.data(), nao2, _nao);
      MMatrixXcd       Ym_2(Y1.data(), _nao, nao2);
      MMatrixXcd       Xm(X.data(), 1, nao3);
      MMatrixXcd       Vm(vcijkl.data(), nao3, _nao);
      MMatrixXcd       Vmx(vxcijkl.data(), nao3, _nao);

      // Loop over tau indices
#pragma omp for
      for (size_t t = tau_offset, ttt = 0; ttt < ntau_local; ++t, ++ttt) {
        int shift = t * _ns * _ink * nao2 + is * _ink * nao2 + momshift;
        int tt    = _nts - t - 1;
        // initialize Green's functions
        for (size_t isp = 0; isp < _ns; ++isp) {
          for (size_t q0 = 0; q0 < _nao; ++q0) {
            for (size_t p0 = 0; p0 < _nao; ++p0) {
              G1(q0, p0) = _bz_utils.symmetry().conj_list()[k[1]] == 0 ? Gr_full_tau(t, is, k1_pos, q0, p0)
                                                                       : std::conj(Gr_full_tau(t, is, k1_pos, q0, p0));
              G2(q0, p0) = _bz_utils.symmetry().conj_list()[k[2]] == 0 ? Gr_full_tau(tt, isp, k2_pos, q0, p0)
                                                                       : std::conj(Gr_full_tau(tt, isp, k2_pos, q0, p0));
              G3(q0, p0) = _bz_utils.symmetry().conj_list()[k[3]] == 0 ? Gr_full_tau(t, isp, k3_pos, q0, p0)
                                                                       : std::conj(Gr_full_tau(t, isp, k3_pos, q0, p0));
            }
          }
          for (size_t i = 0; i < _nao; ++i) {
            // pm,k
            MMatrixXcd Sm(Sigma_local.data() + shift + i * _nao, 1, _nao);
            // v1 for direct
            MMatrixXcd vm_1(vijkl.data() + i * nao3, nao2, _nao);
            // Direct diagram
            contraction(nao2, nao3, is == isp, false, G1, G2, G3, Xm_4, Xm_1, Xm_2, Ym_1, Ym_2, vm_1, Xm, Vm, Vmx, Sm);
          }
        }
      }
    }
    statistics.end();
  }

  void gf2_solver::contraction(size_t nao2, size_t nao3, bool eq_spin, bool ew_correct, const Eigen::MatrixXcd& G1,
                               const Eigen::MatrixXcd& G2, const Eigen::MatrixXcd& G3, MMatrixXcd& Xm_4, MMatrixXcd& Xm_1,
                               MMatrixXcd& Xm_2, MMatrixXcd& Ym_1, MMatrixXcd& Ym_2, const MMatrixXcd& vm_1, MMatrixXcd& Xm,
                               MMatrixXcd& Vm, MMatrixXcd& Vxm, MMatrixXcd& Sm) {
    // pm,l = pm,k k,l
    Xm_1.noalias() = (vm_1 * G3);
    // ml,q = ml,p p,q
    Ym_1.noalias() = (Xm_2.transpose() * G1);
    // n,lq = n,m m,lq
    Xm_2.noalias() = G2 * Ym_2;
    // q,nl
    Xm_4.noalias() = Xm_1.transpose();
    // i,j = i,qnl qnl,j
    Sm.noalias() += Xm * Vm;
    // Additional exchange contribution of same spin
    if (eq_spin and !ew_correct) {
      Sm.noalias() += Xm * Vxm;
    }
  }

  void gf2_solver::read_next(const std::array<size_t, 4>& k) {
    // k = (k1_red, k2, k3, k1_red+k3-k2)
    // Read integral for the 2nd-order direct diagram
    _coul_int_c_1->read_integrals(k[0], k[1]);
    _coul_int_c_2->read_integrals(k[2], k[3]);
    // Read integral for the 2nd-order exchange diagram
    _coul_int_c_3->read_integrals(k[0], k[1]);
    _coul_int_c_4->read_integrals(k[2], k[3]);
    _coul_int_x_3->read_integrals(k[0], k[3]);
    _coul_int_x_4->read_integrals(k[2], k[1]);
  }

  void gf2_solver::setup_integrals(const std::array<size_t, 4>& kpts) {
    vijkl.set_zero();
    vcijkl.set_zero();
    vxijkl.set_zero();
    vxcijkl.set_zero();

    ztensor<3> Vx1(_NQ, _nao, _nao);
    ztensor<3> Vx2(_NQ, _nao, _nao);
    ztensor<3> V1(_NQ, _nao, _nao);
    ztensor<3> V2(_NQ, _nao, _nao);
    _coul_int_x_3->symmetrize(Vx1, kpts[0], kpts[3]);
    _coul_int_x_4->symmetrize(Vx2, kpts[2], kpts[1]);
    _coul_int_c_1->symmetrize(V1, kpts[0], kpts[1]);
    _coul_int_c_2->symmetrize(V2, kpts[2], kpts[3]);
    CMMatrixXcd vx1(Vx1.data(), _NQ, _nao * _nao);
    CMMatrixXcd vx2(Vx2.data(), _NQ, _nao * _nao);
    CMMatrixXcd v1(V1.data(), _NQ, _nao * _nao);
    CMMatrixXcd v2(V2.data(), _NQ, _nao * _nao);

    MMatrixXcd  v(vijkl.data(), _nao * _nao, _nao * _nao);
    MMatrixXcd  vx(vxijkl.data(), _nao * _nao, _nao * _nao);
    MMatrixXcd  vxc(vijkl.data(), _nao * _nao, _nao * _nao);
    MMatrixXcd  vc(vijkl.data(), _nao * _nao, _nao * _nao);

    // For direct diagram
    double      prefactor = (_ns == 2) ? 1.0 : 2.0;
    vc                    = prefactor * v1.transpose().conjugate() * v2.conjugate();

#pragma omp parallel for
    for (size_t i = 0; i < _nao; ++i) {
      for (size_t j = 0; j < _nao; ++j) {
        for (size_t k = 0; k < _nao; ++k) {
          for (size_t l = 0; l < _nao; ++l) {
            // v2 for Direct term
            vcijkl(j, k, l, i) = vijkl(i, j, k, l);
          }
        }
      }
    }
    // For exchange diagram
    vxc = vx1.transpose().conjugate() * vx2.conjugate();
#pragma omp parallel for
    for (size_t i = 0; i < _nao; ++i) {
      for (size_t j = 0; j < _nao; ++j) {
        for (size_t k = 0; k < _nao; ++k) {
          for (size_t l = 0; l < _nao; ++l) {
            // v2 for exchange term
            vxcijkl(j, k, l, i) = -vijkl(i, l, k, j);
          }
        }
      }
    }
    // v1 for direct term
    v = v1.transpose() * v2;
  }
}  // namespace green::mbpt
