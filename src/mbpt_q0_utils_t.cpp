/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#include <set>
#include <cmath>
#include "green/mbpt/mbpt_q0_utils_t.h"

namespace green::mbpt {
  void mbpt_q0_utils_t::aux_to_PW_00(ztensor<4> &X_aux, ztensor<2> &X_PW_00, size_t iq) {
    // X_PW_00[dim0, iq] = Aq[iq, NQ].conj() * X_aux[dim0, NQ, NQ] * Aq[iq, NQ]
    size_t dim0 = X_aux.shape()[0];
    //MMatrixX<prec> vm(v.data(), _NQ * _nao, _nao);
    MMatrixXcd AqQ_m(_AqQ.data(), _ink, _NQ);

    for (size_t i = 0; i < dim0; ++i) {
      // X_PW_00_nw = _Aq(NQ).conj() * X_aux_nw(NQ, NQ) * _Aq(NQ)
      X_PW_00(i, iq) = (AqQ_m.row(iq).conjugate() * matrix(X_aux(i, 0)) * AqQ_m.transpose().col(iq))(0,0);
    }
  }

  void mbpt_q0_utils_t::check_AqQ() {
    MMatrixXcd AqQ_m(_AqQ.data(), _ink, _NQ);
    for (size_t q = 0; q < _ink; ++q) {
      std::complex<double> identity = AqQ_m.row(q).conjugate() * AqQ_m.transpose().col(q);
      if (q != 0) std::cout << "identity = " << identity << std::endl;
    }
  }

  std::complex<double> mbpt_q0_utils_t::extrapolate_q0(std::complex<double> *X, size_t fit_order, double q_max, bool debug) {
    // Find unique q_abs with q_abs <= q_max
    std::vector<int> q_indices = filter_q_abs(q_max);
    size_t num_sample = q_indices.size();
    if (debug) std::cout << "Will use " << num_sample << " points to extrapolate q->0 limit:" << std::endl;
    std::vector<double> X_filtered(q_indices.size());
    std::vector<double> q_filtered(q_indices.size());
    for (int i = 0; i < q_indices.size(); ++i) {
      size_t idx = q_indices[i];
      X_filtered[i] = X[idx].real();
      q_filtered[i] = _q_abs[idx];
      if (debug) std::cout << _q_abs[idx] << std::endl;
    }
    // Construct elements with q_abs <= q_max;
    MatrixXd poly_coeffs(fit_order+1, 1);
    polyfit(q_filtered.data(), X_filtered.data(), fit_order, num_sample, poly_coeffs);
    if (debug) {
      std::cout << "Polynomial at the lowest Matsubara frequency: " <<
                poly_coeffs(0, 0) << " + " << poly_coeffs(1, 0) << "q + " << poly_coeffs(2, 0) << "q^2" << std::endl;
    }
    return poly_coeffs(0, 0);
  }

  std::vector<int> mbpt_q0_utils_t::filter_q_abs(double q_max) {
    std::vector<int> q_indices;
    std::set<double> q_set;
    for (int i = 0; i < _q_abs.size(); ++i) {
      if (_q_abs[i] > 0.0 && _q_abs[i] <= q_max) {
        auto insert_pair = q_set.insert(_q_abs[i]);
        if (insert_pair.second) {
          q_indices.push_back(i);
        }
      }
    }
    return q_indices;
  }

  void mbpt_q0_utils_t::polyfit(double *x, double *y, size_t fit_order, size_t num_sample, MatrixXd &c) {
    MatrixXd A(num_sample, fit_order+1);
    MatrixXd b(num_sample, 1);
    for (int i = 0; i < num_sample; ++i) {
      b(i) = y[i];
      A(i, 0) = 1.0;
      for (int j = 1; j <= fit_order; ++j) {
        A(i, j) = pow(x[i], j);
      }
    }

    // Least-squares fit
    //c = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    // Normal equation: A^{T}Ax = A^{T}b
    Eigen::FullPivLU<MatrixXd> lusolver(fit_order+1, fit_order+1);
    MatrixXd ATA = A.transpose() * A;
    MatrixXd ATb = A.transpose() * b;
    c = lusolver.compute(ATA).solve(ATb);
  }

  void mbpt_q0_utils_t::GW_q0_correction(ztensor<2> &eps_inv_wq, ztensor_view<5> &Sigma, ztensor_view<5> &Gtau,
                                         const grids::transformer_t &ft, bool X2C,
                                         size_t myid, size_t intranode_rank, size_t intranode_size, MPI_Win win_Sigma) {
   //MPI_Allreduce(MPI_IN_PLACE, _eps_inv_wq.data(), _eps_inv_wq.size(), MPI_C_DOUBLE_COMPLEX, MPI_SUM, _comm);
   size_t nts  = Sigma.shape()[0];
   size_t nw_b = eps_inv_wq.shape()[0];
   size_t ink  = eps_inv_wq.shape()[1];

   // Extrapolate to q->0 limit
   ztensor<2> eps_q0_inv_w(nw_b, 1);
   size_t fit_order = 2;
   column poly_coeffs(fit_order+1);
   for (int n = 0; n < nw_b; ++n) {
     // eps_q0_inv_w(n,0) = _eps_q_inv_w(n, 1);
     std::complex<double> *eps_inv_wn_q_ptr = eps_inv_wq.data() + n * ink;
     if (n == nw_b/2) {
       eps_q0_inv_w(n, 0) = (!myid)? extrapolate_q0(eps_inv_wn_q_ptr, fit_order, 1.0, true)
                                    : extrapolate_q0(eps_inv_wn_q_ptr, fit_order, 1.0);
     } else {
       eps_q0_inv_w(n, 0) = extrapolate_q0(eps_inv_wn_q_ptr, fit_order, 1.0);
     }
   }

   // Transform to fermionic tau grid
   ztensor<2> eps_q0_inv_t(nts, 1);
   ft.w_b_to_tau_f(eps_q0_inv_w, eps_q0_inv_t);
   // Apply correction term
   if (!X2C) {
     apply_q0_correction(eps_q0_inv_t, Sigma, Gtau, intranode_rank, intranode_size, win_Sigma);
   } else {
     apply_q0_correction_2C(eps_q0_inv_t, Sigma, Gtau, intranode_rank, intranode_size, win_Sigma);
   }
 }

  void mbpt_q0_utils_t::apply_q0_correction(ztensor<2> &eps_q0_inv_t, ztensor_view<5> &Sigma, ztensor_view<5> &G_tau,
                                            size_t intranode_rank, size_t intranode_size, MPI_Win win_Sigma) {
    size_t nts = Sigma.shape()[0];
    size_t ns  = Sigma.shape()[1];
    size_t ink = Sigma.shape()[2];

    MPI_Win_fence(0, win_Sigma);
    // Delta = -madelung * epsilon_inv_q0 * S * G * S
    for (size_t t = intranode_rank; t < nts; t+=intranode_size) {
      for (size_t s = 0; s < ns; ++s) {
        for (size_t k = 0; k < ink; ++k) {
          matrix(Sigma(t, s, k)) -= _madelung * eps_q0_inv_t(t, 0) *
                                           matrix(_S_k(s, k)) * matrix(G_tau(t, s, k)) * matrix(_S_k(s, k));
        }
      }
    }
    MPI_Win_fence(0, win_Sigma);
  }

  void mbpt_q0_utils_t::apply_q0_correction_2C(ztensor<2> &eps_q0_inv_t, ztensor_view<5> &Sigma, ztensor_view<5> &G_tau,
                                               size_t intranode_rank, size_t intranode_size, MPI_Win win_Sigma) {
    size_t nts = Sigma.shape()[0];
    size_t ns  = Sigma.shape()[1];
    size_t ink = Sigma.shape()[2];
    size_t nso = Sigma.shape()[3];
    size_t nao = nso / 2;

    MPI_Win_fence(0, win_Sigma);
    // Delta = -madelung * epsilon_inv_q0 * S * G * S
    MatrixXcd off_spin_blk_buffer(nao, nao);
    for (size_t t = intranode_rank; t < nts; t+=intranode_size) {
      for (size_t k = 0; k < ink; ++k) {
        MMatrixXcd Sigmam_nso(Sigma.data() + t*ink*nso*nso + k*nso*nso, nso, nso);
        MMatrixXcd Gm_nso(G_tau.data() + t*ink*nso*nso + k*nso*nso, nso, nso);
        CMMatrixXcd Sm_nso(_S_k.data() + k*nso*nso, nso, nso);
        MatrixXcd Sm_nao = Sm_nso.block(0, 0, nao, nao);
        for (size_t s = 0; s < 3; ++s) {
          if (s == 0) {
            // alpha-alpha
            Sigmam_nso.block(0, 0, nao, nao) -= _madelung * eps_q0_inv_t(t, 0) *
                                                  Sm_nao * Gm_nso.block(0, 0, nao, nao) * Sm_nao;
          } else if (s == 1) {
            // beta-beta
            Sigmam_nso.block(nao, nao, nao, nao) -= _madelung * eps_q0_inv_t(t, 0) *
                                                        Sm_nao * Gm_nso.block(nao, nao, nao, nao) * Sm_nao;
          } else {
            // alpha-beta and beta-alpha
            off_spin_blk_buffer = _madelung * eps_q0_inv_t(t, 0) *
                                  Sm_nao * Gm_nso.block(0, nao, nao, nao) * Sm_nao;
            Sigmam_nso.block(0, nao, nao, nao) -= off_spin_blk_buffer;
            Sigmam_nso.block(nao, 0, nao, nao) -= off_spin_blk_buffer.transpose().conjugate();
          }
        }
      }
    }
    MPI_Win_fence(0, win_Sigma);
  }


}
