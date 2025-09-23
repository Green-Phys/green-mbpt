/*
 * Copyright (c) 2024 University of Michigan
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

#include "green/mbpt/kernels.h"

namespace green::mbpt::kernels {
  ztensor<4> hf_scalar_cpu_kernel::solve(const ztensor<4>& dm, const utils::mpi_context& ctx) {
    statistics.start("Hartree-Fock");
    ztensor<4> new_Fock(_ns, _ink, _nao, _nao);
    new_Fock.set_zero();
    {
      df_integral_t coul_int1(_hf_path, _nao, _NQ, _bz_utils, ctx);
      size_t        NQ_local = _NQ / ctx.node_size;
      NQ_local += (_NQ % ctx.node_size > ctx.node_rank) ? 1 : 0;
      size_t NQ_offset = NQ_local * ctx.node_rank +
                         ((_NQ % ctx.node_size > ctx.node_rank) ? 0 : (_NQ % ctx.node_size));
      NQ_offset = (NQ_offset >= _NQ) ? 0 : NQ_offset;
      statistics.start("Direct");
      // Direct diagram
      MatrixXcd  X1(_nao, _nao);
      ztensor<3> v(NQ_local, _nao, _nao);
      ztensor<2> upper_Coul(_NQ, 1);
      MMatrixXcd X1m(X1.data(), _nao * _nao, 1);
      MMatrixXcd vm(v.data(), NQ_local, _nao * _nao);
      MMatrixXcd upper_Coul_m(upper_Coul.data() + NQ_offset, NQ_local, 1);
      for (int ikps = ctx.internode_rank; ikps < _ink * _ns; ikps += ctx.internode_size) {
        int is    = ikps % _ns;
        int ikp   = ikps / _ns;
        int kp_ir = _bz_utils.symmetry().full_point(ikp);
        statistics.start("Read Coulomb Up");
        coul_int1.read_integrals(kp_ir, kp_ir);
        statistics.end();
        if(NQ_local > 0) {
          coul_int1.symmetrize(v, kp_ir, kp_ir, NQ_offset, NQ_local);

          X1 = CMMatrixXcd(dm.data() + is * _ink * _nao * _nao + ikp * _nao * _nao, _nao, _nao);
          X1 = X1.transpose().eval();
          // (Q, 1) = (Q, ab) * (ab, 1)
          upper_Coul_m += _bz_utils.symmetry().weight()[kp_ir] * vm * X1m;
        }
      }
      statistics.start("Reduce Direct");
      MPI_Allreduce(MPI_IN_PLACE, upper_Coul.data(), upper_Coul.size(), MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, ctx.global);
      statistics.end();

      upper_Coul /= double(_nk);
      for (int ii = ctx.internode_rank; ii < _ink * _ns; ii += ctx.internode_size) {
        int is   = ii / _ink;
        int ik   = ii % _ink;
        int k_ir = _bz_utils.symmetry().full_point(ik);
        statistics.start("Read Coulomb Low");
        coul_int1.read_integrals(k_ir, k_ir);
        statistics.end();
        if(NQ_local > 0) {
          coul_int1.symmetrize(v, k_ir, k_ir, NQ_offset, NQ_local);

          MMatrixXcd Fm(new_Fock.data() + is * _ink * _nao * _nao + ik * _nao * _nao, 1, _nao * _nao);
          // (1, ij) = (1, Q) * (Q, ij)
          Fm += upper_Coul_m.transpose() * vm;
        }
      }
      statistics.end();

      // Exchange diagram
      ztensor<3> Y(NQ_local, _nao, _nao);
      MMatrixXcd Ym(Y.data(), NQ_local * _nao, _nao);
      MMatrixXcd Ymm(Y.data(), NQ_local, _nao * _nao);

      ztensor<3> Y1(_nao, _nao, NQ_local);
      MMatrixXcd Y1m(Y1.data(), _nao * _nao, NQ_local);
      MMatrixXcd Y1mm(Y1.data(), _nao, _nao * NQ_local);

      MMatrixXcd vmm(v.data(), NQ_local * _nao, _nao);

      ztensor<3> v2(_nao, NQ_local, _nao);
      MMatrixXcd v2m(v2.data(), _nao, NQ_local * _nao);
      MMatrixXcd v2mm(v2.data(), _nao * NQ_local, _nao);
      double     prefactor = (_ns == 2) ? 1.0 : 0.5;
      statistics.start("Exchange");
      for (int ii = ctx.internode_rank; ii < _ink * _ns; ii += ctx.internode_size) {
        int        is   = ii / _ink;
        int        ik   = ii % _ink;
        int        k_ir = _bz_utils.symmetry().full_point(ik);
        MMatrixXcd Fmm(new_Fock.data() + is * _ink * _nao * _nao + ik * _nao * _nao, _nao, _nao);
        for (int ikp = 0; ikp < _nk; ++ikp) {
          int         kp = _bz_utils.symmetry().reduced_point(ikp);
          CMMatrixXcd dmm(dm.data() + is * _ink * _nao * _nao + kp * _nao * _nao, _nao, _nao);
          statistics.start("Read Coulomb Exch");
          coul_int1.read_integrals(k_ir, ikp);
          statistics.end();
          if(NQ_local > 0) {
            // (Q, i, b) or conj(Q, j, a)
            coul_int1.symmetrize(v, k_ir, ikp, NQ_offset, NQ_local);

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
      }
      statistics.end();

      statistics.start("Ewald correction");
      for (int ii = ctx.global_rank; ii < _ns * _ink; ii += ctx.global_size) {
        int         is = ii / _ink;
        int         ik = ii % _ink;
        CMMatrixXcd dmm(dm.data() + is * _ink * _nao * _nao + ik * _nao * _nao, _nao, _nao);
        CMMatrixXcd Sm(_S_k.data() + is * _ink * _nao * _nao + ik * _nao * _nao, _nao, _nao);
        MMatrixXcd  Fm(new_Fock.data() + is * _ink * _nao * _nao + ik * _nao * _nao, _nao, _nao);
        Fm -= prefactor * _madelung * Sm * dmm * Sm;
      }
      statistics.end();
    }
    statistics.start("Reduce Fock");
    utils::allreduce(MPI_IN_PLACE, new_Fock.data(), new_Fock.size(), MPI_C_DOUBLE_COMPLEX, MPI_SUM, ctx.global);
    statistics.end();
    statistics.end();
    statistics.print(ctx.global);
    return new_Fock;
  }

  ztensor<4> hf_x2c_cpu_kernel::solve(const ztensor<4>& dm, const utils::mpi_context& ctx) {
    statistics.start("X2C Hartree-Fock");
    ztensor<4> new_Fock(1, _ink, _nso, _nso);
    new_Fock.set_zero();
    {
      df_integral_t coul_int1(_hf_path, _nao, _NQ, _bz_utils, ctx);

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

      size_t NQ_local = _NQ / ctx.node_size;
      NQ_local += (_NQ % ctx.node_size > ctx.node_rank) ? 1 : 0;
      size_t NQ_offset = NQ_local * ctx.node_rank +
                         ((_NQ % ctx.node_size > ctx.node_rank) ? 0 : (_NQ % ctx.node_size));
      NQ_offset = (NQ_offset >= _NQ) ? 0 : NQ_offset;
      ztensor<3> v(NQ_local, _nao, _nao);
      // Direct diagram
      // if (ctx.global_rank < _ink) {
      statistics.start("X2C direct diagram");
      {
        MatrixXcd  X1(_nao, _nao);
        MMatrixXcd X1m(X1.data(), _nao * _nao, 1);

        MMatrixXcd vm(v.data(), NQ_local, _nao * _nao);

        ztensor<2> upper_Coul(_NQ, 1);
        MMatrixXcd upper_Coul_m(upper_Coul.data() + NQ_offset, NQ_local, 1);
        for (int ikp = ctx.internode_rank; ikp < _ink; ikp += ctx.internode_size) {
        // for (size_t ikp = 0; ikp < _ink; ++ikp) {
          size_t kp_ir = _bz_utils.symmetry().full_point(ikp);

          coul_int1.read_integrals(kp_ir, kp_ir);
          if(NQ_local > 0) {
            coul_int1.symmetrize(v, kp_ir, kp_ir, NQ_offset, NQ_local);

            // Sum of alpha-alpha and beta-beta spin block
            // In the presence of TR symmetry,
            // (dm_aa(-k) + dm_bb(-k)) = (dm_aa(k) + dm_bb(k))*
            X1 = matrix(dm_spblks[0](ikp)) + matrix(dm_spblks[1](ikp));
            X1 = X1.transpose().eval();
            // (Q, 1) = (Q, ab) * (ab, 1)
            // Since vm(k) = vm(-k)* as well, we only need to take care of half of the k-point
            upper_Coul_m += _bz_utils.symmetry().weight()[kp_ir] * vm * X1m;
          }
        }
        statistics.start("Reduce Direct");
        MPI_Allreduce(MPI_IN_PLACE, upper_Coul.data(), upper_Coul.size(), MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, ctx.global);
        statistics.end();
        upper_Coul /= double(_nk);

        MatrixXcd  Fm(1, _nao * _nao);
        MMatrixXcd Fmm(Fm.data(), _nao, _nao);
        for (int ik = ctx.internode_rank; ik < _ink; ik += ctx.internode_size) {
          int k_ir = _bz_utils.symmetry().full_point(ik);

          coul_int1.read_integrals(k_ir, k_ir);
          if(NQ_local > 0) {
            coul_int1.symmetrize(v, k_ir, k_ir, NQ_offset, NQ_local);

            Fm = upper_Coul_m.transpose() * vm;
            MMatrixXcd Fm_nso(new_Fock.data() + ik * _nso * _nso, _nso, _nso);
            Fm_nso.block(0, 0, _nao, _nao) += Fmm;
            Fm_nso.block(_nao, _nao, _nao, _nao) += Fmm;
          }
        }
      }

      statistics.end();

      statistics.start("X2C exchange diagram");
      // Exchange diagram
      ztensor<3> Y(NQ_local, _nao, _nao);
      MMatrixXcd Ym(Y.data(), NQ_local * _nao, _nao);
      MMatrixXcd Ymm(Y.data(), NQ_local, _nao * _nao);

      ztensor<3> Y1(_nao, _nao, NQ_local);
      MMatrixXcd Y1m(Y1.data(), _nao * _nao, NQ_local);
      MMatrixXcd Y1mm(Y1.data(), _nao, _nao * NQ_local);

      MMatrixXcd vmm(v.data(), NQ_local * _nao, _nao);

      ztensor<3> v2(_nao, NQ_local, _nao);
      MMatrixXcd v2m(v2.data(), _nao, NQ_local * _nao);
      MMatrixXcd v2mm(v2.data(), _nao * NQ_local, _nao);

      for (int iks = ctx.internode_rank; iks < 3 * _ink; iks += ctx.internode_size) {
        size_t      ik = iks / 3;
        size_t      is = iks % 3;
        MMatrixXcd  Fm_nso(new_Fock.data() + ik * _nso * _nso, _nso, _nso);
        MatrixXcd   Fm_spblk(_nao, _nao);
        CMMatrixXcd Sm_nso(_S_k.data() + ik * _nso * _nso, _nso, _nso);
        MatrixXcd   S_aa = Sm_nso.block(0, 0, _nao, _nao);
        if (is == 0) {
          // alpha-alpha
          Fm_nso.block(0, 0, _nao, _nao) +=
              compute_exchange(ik, dm_spblks[0], dm_spblks[1], v, coul_int1, Y, Ym, Ymm, Y1, Y1m, Y1mm, vmm, v2, v2m, v2mm, NQ_local, NQ_offset);
          Fm_nso.block(0, 0, _nao, _nao) -= _madelung * S_aa * matrix(dm_spblks[0](ik)) * S_aa;
        } else if (is == 1) {
          // beta-beta
          Fm_nso.block(_nao, _nao, _nao, _nao) +=
              compute_exchange(ik, dm_spblks[1], dm_spblks[0], v, coul_int1, Y, Ym, Ymm, Y1, Y1m, Y1mm, vmm, v2, v2m, v2mm, NQ_local, NQ_offset);
          Fm_nso.block(_nao, _nao, _nao, _nao) -= _madelung * S_aa * matrix(dm_spblks[1](ik)) * S_aa;
        } else if (is == 2) {
          // alpha-beta
          Fm_nso.block(0, _nao, _nao, _nao) +=
              compute_exchange_ab(ik, dm_spblks[2], v, coul_int1, Y, Ym, Ymm, Y1, Y1m, Y1mm, vmm, v2, v2m, v2mm, NQ_local, NQ_offset);
          Fm_nso.block(0, _nao, _nao, _nao) -= _madelung * S_aa * matrix(dm_spblks[2](ik)) * S_aa;
          // beta-alpha
          Fm_nso.block(_nao, 0, _nao, _nao) = Fm_nso.block(0, _nao, _nao, _nao).transpose().conjugate();
        }
      }
      statistics.end();
    }

    statistics.start("Reduce Fock");
    utils::allreduce(MPI_IN_PLACE, new_Fock.data(), new_Fock.size(), MPI_C_DOUBLE_COMPLEX, MPI_SUM, ctx.global);
    statistics.end();
    statistics.end();
    statistics.print(ctx.global);
    return new_Fock;
  }

  MatrixXcd hf_x2c_cpu_kernel::compute_exchange(int ik, ztensor<3>& dm_s1_s2, ztensor<3>& dm_ms1_ms2, ztensor<3>& v,
                                                df_integral_t& coul_int1, ztensor<3>& Y, MMatrixXcd& Ym, MMatrixXcd& Ymm,
                                                ztensor<3>& Y1, MMatrixXcd& Y1m, MMatrixXcd& Y1mm, MMatrixXcd& vmm,
                                                ztensor<3>& v2, MMatrixXcd& v2m, MMatrixXcd& v2mm, size_t NQ_local, size_t NQ_offset) {
    int       k_ir = _bz_utils.symmetry().full_point(ik);
    MatrixXcd Fock = MatrixXcd::Zero(_nao, _nao);

    for (int ikp = 0; ikp < _nk; ++ikp) {
      int kp = _bz_utils.symmetry().reduced_point(ikp);

      coul_int1.read_integrals(k_ir, ikp);
      if(NQ_local > 0) {
        // (Q, i, b) or conj(Q, j, a)
        coul_int1.symmetrize(v, k_ir, ikp, NQ_offset, NQ_local);

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
    }
    return Fock;
  }

  MatrixXcd hf_x2c_cpu_kernel::compute_exchange_ab(int ik, ztensor<3>& dm_ab, ztensor<3>& v, df_integral_t& coul_int1,
                                                   ztensor<3>& Y, MMatrixXcd& Ym, MMatrixXcd& Ymm, ztensor<3>& Y1,
                                                   MMatrixXcd& Y1m, MMatrixXcd& Y1mm, MMatrixXcd& vmm, ztensor<3>& v2,
                                                   MMatrixXcd& v2m, MMatrixXcd& v2mm, size_t NQ_local, size_t NQ_offset) {
    int       k_ir = _bz_utils.symmetry().full_point(ik);
    MatrixXcd Fock = MatrixXcd::Zero(_nao, _nao);

    for (int ikp = 0; ikp < _nk; ++ikp) {
      int kp = _bz_utils.symmetry().reduced_point(ikp);

      coul_int1.read_integrals(k_ir, ikp);
      if(NQ_local > 0) {
        // (Q, i, b) or conj(Q, j, a)
        coul_int1.symmetrize(v, k_ir, ikp, NQ_offset, NQ_local);

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
    }
    return Fock;
  }
}