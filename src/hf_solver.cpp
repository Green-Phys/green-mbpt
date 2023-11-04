/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#include "green/mbpt/hf_solver.h"

namespace green::mbpt {
  void hf_solver::solve(utils::shared_object<ztensor<5>>& g, ztensor<4>& sigma1, utils::shared_object<ztensor<5>>& sigma_tau) {
    std::array<size_t, 4> shape;
    std::copy(g.object().shape().begin() + 1, g.object().shape().end(), shape.begin());
    ztensor<4> dm(shape);
    dm << g.object()(g.object().shape()[0] - 1);
    dm *= (_ns == 2 or _X2C) ? -1.0 : -2.0;
    if (_X2C) {
      sigma1 << solve_HF_2c(dm);
    } else {
      sigma1 << solve_HF_scalar(dm);
    }
  }
  ztensor<4> hf_solver::solve_HF_scalar(const ztensor<4>& dm) {
    ztensor<4> new_Fock(_ns, _ink, _nao, _nao);
    new_Fock.set_zero();
    if (utils::context.global_rank < _ink * _ns) {
      int           hf_nprocs = (utils::context.global_size > _ink * _ns) ? _ink * _ns : utils::context.global_size;

      df_integral_t coul_int1(_hf_path, _nao, _NQ, _bz_utils);

      // Direct diagram
      MatrixXcd     X1(_nao, _nao);
      ztensor<3>    v(_NQ, _nao, _nao);
      ztensor<2>    upper_Coul(_NQ, 1);
      MMatrixXcd    X1m(X1.data(), _nao * _nao, 1);
      MMatrixXcd    vm(v.data(), _NQ, _nao * _nao);
      for (int ikps = 0; ikps < _ink * _ns; ++ikps) {
        int is    = ikps % _ns;
        int ikp   = ikps / _ns;
        int kp_ir = _bz_utils.symmetry().full_point(ikp);

        coul_int1.read_integrals(kp_ir, kp_ir);
        coul_int1.symmetrize(v, kp_ir, kp_ir);

        X1 = CMMatrixXcd(dm.data() + is * _ink * _nao * _nao + ikp * _nao * _nao, _nao, _nao);
        X1 = X1.transpose().eval();
        // (Q, 1) = (Q, ab) * (ab, 1)
        matrix(upper_Coul) += _bz_utils.symmetry().weight()[kp_ir] * vm * X1m;
      }
      upper_Coul /= double(_nk);

      for (int ii = utils::context.global_rank; ii < _ink * _ns; ii += hf_nprocs) {
        int is   = ii / _ink;
        int ik   = ii % _ink;
        int k_ir = _bz_utils.symmetry().full_point(ik);

        coul_int1.read_integrals(k_ir, k_ir);
        coul_int1.symmetrize(v, k_ir, k_ir);

        MMatrixXcd Fm(new_Fock.data() + is * _ink * _nao * _nao + ik * _nao * _nao, 1, _nao * _nao);
        // (1, ij) = (1, Q) * (Q, ij)
        Fm += matrix(upper_Coul).transpose() * vm;
      }

      // Exchange diagram
      ztensor<3> Y(_NQ, _nao, _nao);
      MMatrixXcd Ym(Y.data(), _NQ * _nao, _nao);
      MMatrixXcd Ymm(Y.data(), _NQ, _nao * _nao);

      ztensor<3> Y1(_nao, _nao, _NQ);
      MMatrixXcd Y1m(Y1.data(), _nao * _nao, _NQ);
      MMatrixXcd Y1mm(Y1.data(), _nao, _nao * _NQ);

      MMatrixXcd vmm(v.data(), _NQ * _nao, _nao);

      ztensor<3> v2(_nao, _NQ, _nao);
      MMatrixXcd v2m(v2.data(), _nao, _NQ * _nao);
      MMatrixXcd v2mm(v2.data(), _nao * _NQ, _nao);
      double     prefactor = (_ns == 2) ? 1.0 : 0.5;

      for (int ii = utils::context.global_rank; ii < _ink * _ns; ii += hf_nprocs) {
        int        is   = ii / _ink;
        int        ik   = ii % _ink;
        int        k_ir = _bz_utils.symmetry().full_point(ik);
        MMatrixXcd Fmm(new_Fock.data() + is * _ink * _nao * _nao + ik * _nao * _nao, _nao, _nao);
        for (int ikp = 0; ikp < _nk; ++ikp) {
          int         kp = _bz_utils.symmetry().reduced_point(ikp);
          CMMatrixXcd dmm(dm.data() + is * _ink * _nao * _nao + kp * _nao * _nao, _nao, _nao);

          coul_int1.read_integrals(k_ir, ikp);
          // (Q, i, b) or conj(Q, j, a)
          coul_int1.symmetrize(v, k_ir, ikp);

          // (Qi, a) = (Qi, b) * (b, a)
          if (_bz_utils.symmetry().conj_list()[ikp] == 0) {
            Ym = vmm * dmm;
          } else {
            Ym = vmm * dmm.conjugate();
          }
          // (ia, Q)
          Y1m = Ymm.transpose();
          // (a, Qj). (Q, j, a) -> (a, Q, j)*
          v2m = vmm.conjugate().transpose();
          // (i, j) = (i, aQ) * (aQ, j)
          Fmm -= prefactor * Y1mm * v2mm / double(_nk);
        }
      }

      for (int ii = utils::context.global_rank; ii < _ns * _ink; ii += hf_nprocs) {
        int         is = ii / _ink;
        int         ik = ii % _ink;
        CMMatrixXcd dmm(dm.data() + is * _ink * _nao * _nao + ik * _nao * _nao, _nao, _nao);
        CMMatrixXcd Sm(_S_k.data() + is * _ink * _nao * _nao + ik * _nao * _nao, _nao, _nao);
        MMatrixXcd  Fm(new_Fock.data() + is * _ink * _nao * _nao + ik * _nao * _nao, _nao, _nao);
        Fm -= prefactor * _madelung * Sm * dmm * Sm;
      }
    }

    utils::allreduce(MPI_IN_PLACE, new_Fock.data(), new_Fock.size(), MPI_C_DOUBLE_COMPLEX, MPI_SUM, utils::context.global);
    return new_Fock;
  }

  ztensor<4> hf_solver::solve_HF_2c(const ztensor<4>& dm) {
    ztensor<4> new_Fock(1, _ink, _nso, _nso);
    new_Fock.set_zero();
    if (utils::context.global_rank < 3 * _ink) {
      int           hf_nprocs = (utils::context.global_size > 3 * _ink) ? 3 * _ink : utils::context.global_size;

      df_integral_t coul_int1(_hf_path, _nao, _NQ, _bz_utils);

      ztensor<3>    dm_spblks[3]{
          {_ink, _nao, _nao},
          {_ink, _nao, _nao},
          {_ink, _nao, _nao}
      };
      for (int ik = 0; ik < _ink; ++ik) {
        CMMatrixXcd dmm(dm.data() + ik * _nso * _nso, _nso, _nso);
        // alpha-alpha
        matrix(dm_spblks[0](ik)) = dmm.block(0, 0, _nao, _nao);
        // beta-beta
        matrix(dm_spblks[1](ik)) = dmm.block(_nao, _nao, _nao, _nao);
        // alpha-beta
        matrix(dm_spblks[2](ik)) = dmm.block(0, _nao, _nao, _nao);
      }

      ztensor<3> v(_NQ, _nao, _nao);
      // Direct diagram
      if (utils::context.global_rank < _ink) {
        MatrixXcd  X1(_nao, _nao);
        MMatrixXcd X1m(X1.data(), _nao * _nao, 1);

        MMatrixXcd vm(v.data(), _NQ, _nao * _nao);

        ztensor<2> upper_Coul(_NQ, 1);
        for (size_t ikp = 0; ikp < _ink; ++ikp) {
          size_t kp_ir = _bz_utils.symmetry().full_point(ikp);

          coul_int1.read_integrals(kp_ir, kp_ir);
          coul_int1.symmetrize(v, kp_ir, kp_ir);

          // Sum of alpha-alpha and beta-beta spin block
          // In the presence of TR symmetry,
          // (dm_aa(-k) + dm_bb(-k)) = (dm_aa(k) + dm_bb(k))*
          X1 = matrix(dm_spblks[0](ikp)) + matrix(dm_spblks[1](ikp));
          X1 = X1.transpose().eval();
          // (Q, 1) = (Q, ab) * (ab, 1)
          // Since vm(k) = vm(-k)* as well, we only need to take care of half of the k-point
          matrix(upper_Coul) += _bz_utils.symmetry().weight()[kp_ir] * vm * X1m;
        }
        upper_Coul /= double(_nk);

        MatrixXcd  Fm(1, _nao * _nao);
        MMatrixXcd Fmm(Fm.data(), _nao, _nao);
        for (int ik = utils::context.global_rank; ik < _ink; ik += hf_nprocs) {
          int k_ir = _bz_utils.symmetry().full_point(ik);

          coul_int1.read_integrals(k_ir, k_ir);
          coul_int1.symmetrize(v, k_ir, k_ir);

          Fm = matrix(upper_Coul).transpose() * vm;
          MMatrixXcd Fm_nso(new_Fock.data() + ik * _nso * _nso, _nso, _nso);
          Fm_nso.block(0, 0, _nao, _nao) += Fmm;
          Fm_nso.block(_nao, _nao, _nao, _nao) += Fmm;
        }
      }

      // Exchange diagram
      ztensor<3> Y(_NQ, _nao, _nao);
      MMatrixXcd Ym(Y.data(), _NQ * _nao, _nao);
      MMatrixXcd Ymm(Y.data(), _NQ, _nao * _nao);

      ztensor<3> Y1(_nao, _nao, _NQ);
      MMatrixXcd Y1m(Y1.data(), _nao * _nao, _NQ);
      MMatrixXcd Y1mm(Y1.data(), _nao, _nao * _NQ);

      MMatrixXcd vmm(v.data(), _NQ * _nao, _nao);

      ztensor<3> v2(_nao, _NQ, _nao);
      MMatrixXcd v2m(v2.data(), _nao, _NQ * _nao);
      MMatrixXcd v2mm(v2.data(), _nao * _NQ, _nao);

      for (size_t iks = utils::context.global_rank; iks < 3 * _ink; iks += hf_nprocs) {
        size_t      ik = iks / 3;
        size_t      is = iks % 3;
        MMatrixXcd  Fm_nso(new_Fock.data() + ik * _nso * _nso, _nso, _nso);
        MatrixXcd   Fm_spblk(_nao, _nao);
        CMMatrixXcd Sm_nso(_S_k.data() + ik * _nso * _nso, _nso, _nso);
        MatrixXcd   S_aa = Sm_nso.block(0, 0, _nao, _nao);
        if (is == 0) {
          // alpha-alpha
          Fm_nso.block(0, 0, _nao, _nao) +=
              compute_exchange(ik, dm_spblks[0], dm_spblks[1], v, coul_int1, Y, Ym, Ymm, Y1, Y1m, Y1mm, vmm, v2, v2m, v2mm);
          Fm_nso.block(0, 0, _nao, _nao) -= _madelung * S_aa * matrix(dm_spblks[0](ik)) * S_aa;
        } else if (is == 1) {
          // beta-beta
          Fm_nso.block(_nao, _nao, _nao, _nao) +=
              compute_exchange(ik, dm_spblks[1], dm_spblks[0], v, coul_int1, Y, Ym, Ymm, Y1, Y1m, Y1mm, vmm, v2, v2m, v2mm);
          Fm_nso.block(_nao, _nao, _nao, _nao) -= _madelung * S_aa * matrix(dm_spblks[1](ik)) * S_aa;
        } else if (is == 2) {
          // alpha-beta
          Fm_nso.block(0, _nao, _nao, _nao) +=
              compute_exchange_ab(ik, dm_spblks[2], v, coul_int1, Y, Ym, Ymm, Y1, Y1m, Y1mm, vmm, v2, v2m, v2mm);
          Fm_nso.block(0, _nao, _nao, _nao) -= _madelung * S_aa * matrix(dm_spblks[2](ik)) * S_aa;
          // beta-alpha
          Fm_nso.block(_nao, 0, _nao, _nao) = Fm_nso.block(0, _nao, _nao, _nao).transpose().conjugate();
        }
      }
    }

    utils::allreduce(MPI_IN_PLACE, new_Fock.data(), new_Fock.size(), MPI_C_DOUBLE_COMPLEX, MPI_SUM, utils::context.global);
    return new_Fock;
  }

  MatrixXcd hf_solver::compute_exchange(int ik, ztensor<3>& dm_s1_s2, ztensor<3>& dm_ms1_ms2, ztensor<3>& v,
                                        df_integral_t& coul_int1, ztensor<3>& Y, MMatrixXcd& Ym, MMatrixXcd& Ymm, ztensor<3>& Y1,
                                        MMatrixXcd& Y1m, MMatrixXcd& Y1mm, MMatrixXcd& vmm, ztensor<3>& v2, MMatrixXcd& v2m,
                                        MMatrixXcd& v2mm) {
    int       k_ir = _bz_utils.symmetry().full_point(ik);
    MatrixXcd Fock = MatrixXcd::Zero(_nao, _nao);

    for (int ikp = 0; ikp < _nk; ++ikp) {
      int kp = _bz_utils.symmetry().reduced_point(ikp);

      coul_int1.read_integrals(k_ir, ikp);
      // (Q, i, b) or conj(Q, j, a)
      coul_int1.symmetrize(v, k_ir, ikp);

      // (Qi, a) = (Qi, b) * (b, a)
      if (_bz_utils.symmetry().conj_list()[ikp] == 0) {
        CMMatrixXcd dmm(dm_s1_s2.data() + kp * _nao * _nao, _nao, _nao);
        Ym = vmm * dmm;
      } else {
        CMMatrixXcd dmm(dm_ms1_ms2.data() + kp * _nao * _nao, _nao, _nao);
        Ym = vmm * dmm.conjugate();
      }
      // (ia, Q)
      Y1m = Ymm.transpose();
      // (a, Qj)
      v2m = vmm.conjugate().transpose();
      // (i, j) = (i, aQ) * (aQ, j)
      Fock -= Y1mm * v2mm / double(_nk);
    }
    return Fock;
  }

  MatrixXcd hf_solver::compute_exchange_ab(int ik, ztensor<3>& dm_ab, ztensor<3>& v, df_integral_t& coul_int1, ztensor<3>& Y,
                                           MMatrixXcd& Ym, MMatrixXcd& Ymm, ztensor<3>& Y1, MMatrixXcd& Y1m, MMatrixXcd& Y1mm,
                                           MMatrixXcd& vmm, ztensor<3>& v2, MMatrixXcd& v2m, MMatrixXcd& v2mm) {
    int       k_ir = _bz_utils.symmetry().full_point(ik);
    MatrixXcd Fock = MatrixXcd::Zero(_nao, _nao);

    for (int ikp = 0; ikp < _nk; ++ikp) {
      int kp = _bz_utils.symmetry().reduced_point(ikp);

      coul_int1.read_integrals(k_ir, ikp);
      // (Q, i, b) or conj(Q, j, a)
      coul_int1.symmetrize(v, k_ir, ikp);

      // (Qi, a) = (Qi, b) * (b, a)
      if (_bz_utils.symmetry().conj_list()[ikp] == 0) {
        CMMatrixXcd dmm(dm_ab.data() + kp * _nao * _nao, _nao, _nao);
        Ym = vmm * dmm;
      } else {
        CMMatrixXcd dmm(dm_ab.data() + kp * _nao * _nao, _nao, _nao);
        Ym = vmm * (-1.0) * dmm.transpose();
      }
      // (ia, Q)
      Y1m = Ymm.transpose();
      // (a, Qj)
      v2m = vmm.conjugate().transpose();
      // (i, j) = (i, aQ) * (aQ, j)
      Fock -= Y1mm * v2mm / double(_nk);
    }
    return Fock;
  }
}  // namespace green::mbpt
