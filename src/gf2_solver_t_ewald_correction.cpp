/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#include "green/mbpt/gf2_solver.h"

namespace green::mbpt {
  void gf2_solver::compute_2nd_exch_correction(size_t tau_offset, size_t ntau_local, const ztensor<5>& Gr_full_tau) {
    // compute Ewald correction to the second order
    size_t     nao2 = _nao * _nao;
    size_t     nao3 = _nao * _nao * _nao;
    MatrixXcd  G1(_nao, _nao);
    MatrixXcd  G2(_nao, _nao);
    MatrixXcd  G3(_nao, _nao);
    ztensor<3> X(_nao, _nao, _nao);
    MMatrixXcd Xm_4(X.data(), _nao, nao2);
    ztensor<3> X1(_nao, _nao, _nao);
    MMatrixXcd Xm_1(X1.data(), nao2, _nao);
    MMatrixXcd Xm_2(X1.data(), _nao, nao2);
    ztensor<3> Y1(_nao, _nao, _nao);
    MMatrixXcd Ym_1(Y1.data(), nao2, _nao);
    MMatrixXcd Ym_2(Y1.data(), _nao, nao2);
    MMatrixXcd Xm(X.data(), 1, nao3);
    // TODO Check this
    MMatrixXcd Vm(vcijkl.data(), nao3, _nao);
    ewald_2nd_order_0_0(tau_offset, ntau_local, Gr_full_tau, G1, G2, G3, Xm_4, Xm_1, Xm_2, Ym_1, Ym_2, Xm, Vm);
    ewald_2nd_order_0_1(tau_offset, ntau_local, Gr_full_tau, G1, G2, G3, Xm_4, Xm_1, Xm_2, Ym_1, Ym_2, Xm, Vm);
    ewald_2nd_order_1_0(tau_offset, ntau_local, Gr_full_tau, G1, G2, G3, Xm_4, Xm_1, Xm_2, Ym_1, Ym_2, Xm, Vm);
  }

  void gf2_solver::ewald_2nd_order_0_0(size_t tau_offset, size_t ntau_local, const ztensor<5>& Gr_full_tau, MatrixXcd& G1, MatrixXcd& G2, MatrixXcd& G3,
                                       MMatrixXcd& Xm_4, MMatrixXcd& Xm_1, MMatrixXcd& Xm_2, MMatrixXcd& Ym_1, MMatrixXcd& Ym_2,
                                       MMatrixXcd& Xm, MMatrixXcd& Vm) {
    size_t nao2 = _nao * _nao;
    size_t nao3 = nao2 * _nao;
    _coul_int_c_1->reset();
    _coul_int_c_2->reset();
    _coul_int_c_3->reset();
    _coul_int_c_4->reset();
    _coul_int_x_3->reset();
    _coul_int_x_4->reset();
    int old_k1 = -1;
    // start main loop
    // execution will proceed while current point is non-negative
    for (size_t k1i = utils::context.internode_rank; k1i < _ink * _nao; k1i += utils::context.internode_size) {
      size_t k1_pos   = k1i / _nao;
      int    k1_red   = _bz_utils.symmetry().full_point(k1_pos);
      size_t i        = k1i % _nao;
      size_t momshift = k1_pos * nao2;
      // read next part of integrals
      if (old_k1 != k1_red) {
        read_next_correction_0_0(k1_red);
        old_k1 = k1_red;
      }
      for(size_t t = tau_offset, lt = 0; lt< ntau_local; ++lt, ++t) {
        size_t tt = _nts - t - 1;
        for (size_t s = 0; s < _ns; ++s) {
          size_t shift = t * _ns * _ink * nao2 + s * _ink * nao2 + momshift;
          // initialize Green's functions
          G1           = extract_G_tau_k(Gr_full_tau, t, k1_pos, k1_red, s);
          G2           = extract_G_tau_k(Gr_full_tau, tt, k1_pos, k1_red, s);
          G3           = extract_G_tau_k(Gr_full_tau, t, k1_pos, k1_red, s);
          MMatrixXcd Sm(Sigma_local.data() + shift + i * _nao, 1, _nao);
          MMatrixXcd vm_1(vijkl.data() + i * nao3, nao2, _nao);
          contraction(nao2, nao3, false, true, G1, G2, G3, Xm_4, Xm_1, Xm_2, Ym_1, Ym_2, vm_1, Xm, Vm, Vm, Sm);
        }
      }
    }
    //MPI_Barrier(utils::context.global);
  }

  void gf2_solver::ewald_2nd_order_0_1(size_t tau_offset, size_t ntau_local, const ztensor<5>& Gr_full_tau, MatrixXcd& G1, MatrixXcd& G2, MatrixXcd& G3,
                                       MMatrixXcd& Xm_4, MMatrixXcd& Xm_1, MMatrixXcd& Xm_2, MMatrixXcd& Ym_1, MMatrixXcd& Ym_2,
                                       MMatrixXcd& Xm, MMatrixXcd& Vm) {
    size_t nao2 = _nao * _nao;
    size_t nao3 = nao2 * _nao;
    _coul_int_c_1->reset();
    _coul_int_c_2->reset();
    _coul_int_c_3->reset();
    _coul_int_c_4->reset();
    _coul_int_x_3->reset();
    _coul_int_x_4->reset();
    int old_k1 = -1;
    int old_k2 = -1;
    // start main loop
    // execution will proceed while current point is non-negative
    for (size_t k1k2 = utils::context.internode_rank; k1k2 < _ink * _nk; k1k2 += utils::context.internode_size) {
      // k1k2 = k1 * _nk + k2
      size_t k1_pos   = k1k2 / (_nk);
      int    k1_red   = _bz_utils.symmetry().full_point(k1_pos);

      int    k2_red   = (k1k2) % _nk;
      size_t k2_pos   = _bz_utils.symmetry().reduced_point(k2_red);
      size_t momshift = k1_pos * nao2;
      // read next part of integrals
      if (old_k1 != k1_red || old_k2 != k2_red) {
        read_next_correction_0_1(k1_red, k2_red);
        old_k1 = k1_red;
        old_k2 = k2_red;
      }
      for(size_t t = tau_offset, lt = 0; lt< ntau_local; ++lt, ++t) {
        size_t tt = _nts - t - 1;
        for (size_t s = 0; s < _ns; ++s) {
          size_t shift = t * _ns * _ink * nao2 + s * _ink * nao2 + momshift;
          // initialize Green's functions
          G1           = extract_G_tau_k(Gr_full_tau, t, k1_pos, k1_red, s);
          G2           = extract_G_tau_k(Gr_full_tau, tt, k2_pos, k2_red, s);
          G3           = extract_G_tau_k(Gr_full_tau, t, k2_pos, k2_red, s);
          for (size_t i = 0; i < _nao; ++i) {
            MMatrixXcd Sm(Sigma_local.data() + shift + i * _nao, 1, _nao);
            MMatrixXcd vm_1(vijkl.data() + i * nao3, nao2, _nao);
            contraction(nao2, nao3, false, true, G1, G2, G3, Xm_4, Xm_1, Xm_2, Ym_1, Ym_2, vm_1, Xm, Vm, Vm, Sm);
          }
        }
      }
    }
//    MPI_Barrier(utils::context.global);
  }

  void gf2_solver::ewald_2nd_order_1_0(size_t tau_offset, size_t ntau_local, const ztensor<5>& Gr_full_tau, MatrixXcd& G1, MatrixXcd& G2, MatrixXcd& G3,
                                       MMatrixXcd& Xm_4, MMatrixXcd& Xm_1, MMatrixXcd& Xm_2, MMatrixXcd& Ym_1, MMatrixXcd& Ym_2,
                                       MMatrixXcd& Xm, MMatrixXcd& Vm) {
    size_t nao2   = _nao * _nao;
    size_t nao3   = nao2 * _nao;
    int    old_k1 = -1;
    int    old_k2 = -1;
    _coul_int_c_1->reset();
    _coul_int_c_2->reset();
    _coul_int_c_3->reset();
    _coul_int_c_4->reset();
    _coul_int_x_3->reset();
    _coul_int_x_4->reset();
    // start main loop
    for (size_t k1k2 = utils::context.internode_rank; k1k2 < _ink * _nk; k1k2 += utils::context.internode_size) {
      // k1k2 = k1 * _nk + k2
      size_t k1_pos   = k1k2 / (_nk);
      int    k1_red   = _bz_utils.symmetry().full_point(k1_pos);
      int    k2_red   = k1k2 % _nk;
      size_t k2_pos   = _bz_utils.symmetry().reduced_point(k2_red);
      size_t momshift = k1_pos * nao2;
      // read next part of integrals
      if (old_k1 != k1_red || old_k2 != k2_red) {
        read_next_correction_1_0(k1_red, k2_red);
        old_k1 = k1_red;
        old_k2 = k2_red;
      }
      for(size_t t = tau_offset, lt = 0; lt< ntau_local; ++lt, ++t) {
        size_t tt = _nts - t - 1;
        for (size_t s = 0; s < _ns; ++s) {
          size_t shift = t * _ns * _ink * nao2 + s * _ink * nao2 + momshift;
          // initialize Green's functions
          G1           = extract_G_tau_k(Gr_full_tau, t, k2_pos, k2_red, s);
          G2           = extract_G_tau_k(Gr_full_tau, tt, k2_pos, k2_red, s);
          G3           = extract_G_tau_k(Gr_full_tau, t, k1_pos, k1_red, s);
          for (size_t i = 0; i < _nao; ++i) {
            MMatrixXcd Sm(Sigma_local.data() + shift + i * _nao, 1, _nao);
            MMatrixXcd vm_1(vijkl.data() + i * nao3, nao2, _nao);
            contraction(nao2, nao3, false, true, G1, G2, G3, Xm_4, Xm_1, Xm_2, Ym_1, Ym_2, vm_1, Xm, Vm, Vm, Sm);
          }
        }
      }
    }
//    MPI_Barrier(utils::context.global);
  }

  void gf2_solver::read_next_correction_0_0(size_t k) {
    _coul_int_c_1->read_correction(k);
    _coul_int_c_2->read_correction(k);
    _coul_int_x_3->read_correction(k);
    _coul_int_x_4->read_correction(k);
    // direct
    CMMatrixXcd vc_1(_coul_int_c_1->v0ij_Q().data(), _NQ, _nao * _nao);
    CMMatrixXcd vc_2(_coul_int_c_2->v0ij_Q().data(), _NQ, _nao * _nao);
    CMMatrixXcd vc_bar_1(_coul_int_c_1->v_bar_ij_Q().data(), _NQ, _nao * _nao);
    CMMatrixXcd vc_bar_2(_coul_int_c_2->v_bar_ij_Q().data(), _NQ, _nao * _nao);
    // exchange
    CMMatrixXcd vx_3(_coul_int_x_3->v0ij_Q().data(), _NQ, _nao * _nao);
    CMMatrixXcd vx_4(_coul_int_x_4->v0ij_Q().data(), _NQ, _nao * _nao);
    CMMatrixXcd vx_bar_3(_coul_int_x_3->v_bar_ij_Q().data(), _NQ, _nao * _nao);
    CMMatrixXcd vx_bar_4(_coul_int_x_4->v_bar_ij_Q().data(), _NQ, _nao * _nao);

    MMatrixXcd  v(vijkl.data(), _nao * _nao, _nao * _nao);
    MMatrixXcd  vx2(vijkl.data(), _nao * _nao, _nao * _nao);
    vijkl.set_zero();
    vcijkl.set_zero();

    vx2 = vx_3.transpose() * vx_4 + vx_3.transpose() * vx_bar_4 + vx_bar_3.transpose() * vx_4;
#pragma omp parallel for
    for (size_t i = 0; i < _nao; ++i) {
      for (size_t j = 0; j < _nao; ++j) {
        for (size_t k = 0; k < _nao; ++k) {
          for (size_t l = 0; l < _nao; ++l) {
            vcijkl(k, l, i, j) = -vijkl(i, j, k, l);
          }
        }
      }
    }
    v = vc_1.transpose() * vc_2 + vc_bar_1.transpose() * vc_2 + vc_1.transpose() * vc_bar_2;
  }

  void gf2_solver::read_next_correction_0_1(size_t k1, size_t k2) {
    _coul_int_c_1->read_correction(k1);
    _coul_int_c_2->read_correction(k2);
    // direct
    CMMatrixXcd vc_1(_coul_int_c_1->v0ij_Q().data(), _NQ, _nao * _nao);
    CMMatrixXcd vc_2(_coul_int_c_2->v0ij_Q().data(), _NQ, _nao * _nao);
    CMMatrixXcd vc_bar_1(_coul_int_c_1->v_bar_ij_Q().data(), _NQ, _nao * _nao);
    CMMatrixXcd vc_bar_2(_coul_int_c_2->v_bar_ij_Q().data(), _NQ, _nao * _nao);
    // exchange
    CMMatrixXcd vx_3(_coul_int_x_3->access_vij_Q(k2,k1), _NQ, _nao * _nao);
    CMMatrixXcd vx_4(_coul_int_x_4->access_vij_Q(k1,k2), _NQ, _nao * _nao);
    MMatrixXcd  v(vijkl.data(), _nao * _nao, _nao * _nao);
    MMatrixXcd  vx2(vijkl.data(), _nao * _nao, _nao * _nao);
    vijkl.set_zero();
    vcijkl.set_zero();

    vx2 = vx_3.transpose() * vx_4;
    _coul_int_x_3->release_vij_Q(k2,k1);
    _coul_int_x_4->release_vij_Q(k1,k2);
//#pragma omp parallel for
    for (size_t i = 0; i < _nao; ++i) {
      for (size_t j = 0; j < _nao; ++j) {
        for (size_t k = 0; k < _nao; ++k) {
          for (size_t l = 0; l < _nao; ++l) {
            vcijkl(k, l, i, j) = -vijkl(i, j, k, l);
          }
        }
      }
    }
    v = vc_1.transpose() * vc_2 + vc_bar_1.transpose() * vc_2 + vc_1.transpose() * vc_bar_2;
  }

  void gf2_solver::read_next_correction_1_0(size_t k1, size_t k2) {
    _coul_int_x_3->read_correction(k1);
    _coul_int_x_4->read_correction(k2);
    // direct
    CMMatrixXcd vc_1(_coul_int_c_1->access_vij_Q(k1,k2), _NQ, _nao * _nao);
    CMMatrixXcd vc_2(_coul_int_c_2->access_vij_Q(k2,k1), _NQ, _nao * _nao);
    // exchange
    CMMatrixXcd vx_3(_coul_int_x_3->v0ij_Q().data(), _NQ, _nao * _nao);
    CMMatrixXcd vx_4(_coul_int_x_4->v0ij_Q().data(), _NQ, _nao * _nao);
    CMMatrixXcd vx_bar_3(_coul_int_x_3->v_bar_ij_Q().data(), _NQ, _nao * _nao);
    CMMatrixXcd vx_bar_4(_coul_int_x_4->v_bar_ij_Q().data(), _NQ, _nao * _nao);
    MMatrixXcd  v(vijkl.data(), _nao * _nao, _nao * _nao);
    MMatrixXcd  vx2(vijkl.data(), _nao * _nao, _nao * _nao);
    vijkl.set_zero();
    vcijkl.set_zero();

    vx2 = vx_3.transpose() * vx_4 + vx_3.transpose() * vx_bar_4 + vx_bar_3.transpose() * vx_4;
#pragma omp parallel for
    for (size_t i = 0; i < _nao; ++i) {
      for (size_t j = 0; j < _nao; ++j) {
        for (size_t k = 0; k < _nao; ++k) {
          for (size_t l = 0; l < _nao; ++l) {
            vcijkl(k, l, i, j) = -vijkl(i, j, k, l);
          }
        }
      }
    }
    v = vc_1.transpose() * vc_2;
    _coul_int_c_1->release_vij_Q(k1,k2);
    _coul_int_c_2->release_vij_Q(k2,k1);
  }
}  // namespace green::mbpt
