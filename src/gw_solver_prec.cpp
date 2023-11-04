/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#ifndef GF2_GW_SOLVER_PREC_H
#define GF2_GW_SOLVER_PREC_H

#include "green/mbpt/gw_solver.h"

namespace green::mbpt {

  template <typename prec>
  void gw_solver::eval_P0_tilde(const std::array<size_t, 4>& k, const G_type& G) {
    // k = (k1, 0, q_ir, k1+q_ir)
    // Link current k-points to the corresponding irreducible one
    // size_t k1 = _bz_utils.index()[k[0]];
    // size_t k1q = _bz_utils.index()[k[3]];

    size_t          tau_batch = (_nts / 2) / _ntauprocs;
    size_t          tau_rest  = (_nts / 2) % _ntauprocs;
    size_t          t_start   = (_tauid < tau_rest) ? _tauid * (tau_batch + 1) : _tauid * tau_batch + tau_rest;
    size_t          t_end     = (_tauid < tau_rest) ? (_tauid + 1) * (tau_batch + 1) : (_tauid + 1) * tau_batch + tau_rest;

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
      for (size_t t = t_start; t < t_end; ++t) {  // Loop over half-tau
        size_t     tt = _nts - t - 1;             // beta - t
        MMatrixXcd P0(_P0_tilde.data() + t * _NQ * _NQ, _NQ, _NQ);
        for (size_t s = _spinid; s < pseudo_ns; s += _nspinprocs) {
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
  void gw_solver::assign_G(size_t k, size_t t, size_t s, const ztensor<5>& G_fermi, MatrixX<prec>& G_k) {
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
  void gw_solver::assign_G_nso(size_t k, size_t t, size_t s1, size_t s2, const ztensor<5>& G_fermi, MatrixX<prec>& G_k) {
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
  void gw_solver::P0_contraction(const MatrixX<prec>& Gb_k1, const MatrixX<prec>& G_k1q, MMatrixX<prec>& vm, MMatrixX<prec>& VVm,
                                 MMatrixX<prec>& VVmm, MMatrixX<prec>& X1m, MMatrixX<prec>& vmm, MMatrixX<prec>& X2m,
                                 MMatrixX<prec>& X1mm, MMatrixX<prec>& X2mm, MMatrixXcd& P0, double& prefactor) {
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

  template <typename prec>
  void gw_solver::eval_selfenergy(const std::array<size_t, 4>& k, const G_type& G_fermi, St_type& Sigma_fermi_s) {
    // k = (k1_ir, q_deg, 0, k1_ir-q_deg)
    // Link to corresponding irreducible k-point
    size_t          k1q_pos     = _bz_utils.symmetry().reduced_point(k[3]);
    size_t          tau_batch   = (_nts) / _ntauprocs;
    size_t          tau_rest    = (_nts) % _ntauprocs;
    size_t          t_start     = (_tauid < tau_rest) ? _tauid * (tau_batch + 1) : _tauid * tau_batch + tau_rest;
    size_t          t_end       = (_tauid < tau_rest) ? (_tauid + 1) * (tau_batch + 1) : (_tauid + 1) * tau_batch + tau_rest;
    auto&           Sigma_fermi = Sigma_fermi_s.object();

    size_t          k1_pos      = _bz_utils.symmetry().reduced_point(k[0]);
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
      for (size_t t = t_start; t < t_end; ++t) {
        MMatrixXcd    P(_P0_tilde.data() + t * _NQ * _NQ, _NQ, _NQ);
        MatrixX<prec> P_sp(_NQ, _NQ);
        P_sp = P.cast<prec>();
        for (size_t s = _spinid; s < pseudo_ns; s += _nspinprocs) {
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
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, Sigma_fermi_s.win());
            Sm.noalias() -= Sigma_ts;
            MPI_Win_unlock(0, Sigma_fermi_s.win());
          } else {
            sigma_shift = t * _ns * _ink * _nso * _nso + 0 * _ink * _nso * _nso + k1_pos * _nso * _nso;
            MMatrixXcd Sm_nso(Sigma_fermi.data() + sigma_shift, _nso, _nso);
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, Sigma_fermi_s.win());
            Sm_nso.block(a * _nao, b * _nao, _nao, _nao) -= Sigma_ts;
            MPI_Win_unlock(0, Sigma_fermi_s.win());
          }
        }
      }
    }
  }

  /**
   * Contraction for evaluating self-energy for given tau and k-point
   */
  template <typename prec>
  void gw_solver::selfenergy_contraction(const std::array<size_t, 4>& k, const MatrixX<prec>& G_k1q, MMatrixX<prec>& vm,
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
  template void gw_solver::eval_P0_tilde<std::complex<float>>(const std::array<size_t, 4>& k, const G_type&);
  template void gw_solver::eval_P0_tilde<std::complex<double>>(const std::array<size_t, 4>& k, const G_type&);
  template void gw_solver::P0_contraction(const MatrixX<std::complex<float>>& Gb_k1, const MatrixX<std::complex<float>>& G_k1q,
                                          MMatrixX<std::complex<float>>& vm, MMatrixX<std::complex<float>>& VVm,
                                          MMatrixX<std::complex<float>>& VVmm, MMatrixX<std::complex<float>>& X1m,
                                          MMatrixX<std::complex<float>>& vmm, MMatrixX<std::complex<float>>& X2m,
                                          MMatrixX<std::complex<float>>& X1mm, MMatrixX<std::complex<float>>& X2mm,
                                          MMatrixXcd& P0, double& prefactor);
  template void gw_solver::P0_contraction(const MatrixX<std::complex<double>>& Gb_k1, const MatrixX<std::complex<double>>& G_k1q,
                                          MMatrixX<std::complex<double>>& vm, MMatrixX<std::complex<double>>& VVm,
                                          MMatrixX<std::complex<double>>& VVmm, MMatrixX<std::complex<double>>& X1m,
                                          MMatrixX<std::complex<double>>& vmm, MMatrixX<std::complex<double>>& X2m,
                                          MMatrixX<std::complex<double>>& X1mm, MMatrixX<std::complex<double>>& X2mm,
                                          MMatrixXcd& P0, double& prefactor);
  template void gw_solver::eval_selfenergy<std::complex<float>>(const std::array<size_t, 4>& k, const G_type&, St_type&);
  template void gw_solver::eval_selfenergy<std::complex<double>>(const std::array<size_t, 4>& k, const G_type&, St_type&);
  template void gw_solver::selfenergy_contraction(const std::array<size_t, 4>& k, const MatrixX<std::complex<float>>& G_k1q,
                                                  MMatrixX<std::complex<float>>& vm, MMatrixX<std::complex<float>>& Y1m,
                                                  MMatrixX<std::complex<float>>& Y1mm, MMatrixX<std::complex<float>>& Y2mm,
                                                  MMatrixX<std::complex<float>>& X2m, MMatrixX<std::complex<float>>& Y2mmm,
                                                  MMatrixX<std::complex<float>>& X2mm, MatrixX<std::complex<float>>& P,
                                                  MatrixXcd& Sm_ts);
  template void gw_solver::selfenergy_contraction(const std::array<size_t, 4>& k, const MatrixX<std::complex<double>>& G_k1q,
                                                  MMatrixX<std::complex<double>>& vm, MMatrixX<std::complex<double>>& Y1m,
                                                  MMatrixX<std::complex<double>>& Y1mm, MMatrixX<std::complex<double>>& Y2mm,
                                                  MMatrixX<std::complex<double>>& X2m, MMatrixX<std::complex<double>>& Y2mmm,
                                                  MMatrixX<std::complex<double>>& X2mm, MatrixX<std::complex<double>>& P,
                                                  MatrixXcd& Sm_ts);
  template void gw_solver::assign_G(size_t k, size_t t, size_t s, const ztensor<5>&, MatrixX<std::complex<double>>& G_k);
  template void gw_solver::assign_G(size_t k, size_t t, size_t s, const ztensor<5>&, MatrixX<std::complex<float>>& G_k);
  template void gw_solver::assign_G_nso(size_t k, size_t t, size_t s1, size_t s2, const ztensor<5>&,
                                        MatrixX<std::complex<double>>& G_k);
  template void gw_solver::assign_G_nso(size_t k, size_t t, size_t s1, size_t s2, const ztensor<5>&,
                                        MatrixX<std::complex<float>>& G_k);
}  // namespace green::mbpt

#endif  // GF2_GW_SOLVER_PREC_H
