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
  ztensor<4> hf_scalar_cpu_kernel::solve(const ztensor<4>& dm) {
    statistics.start("Hartree-Fock");
    ztensor<4> new_Fock(_ns, _ink, _nao, _nao);
    new_Fock.set_zero();
    {
      df_integral_t coul_int1(_hf_path, _nao, _NQ, _bz_utils);
      size_t        NQ_local = _NQ / utils::context().node_size;
      NQ_local += (_NQ % utils::context().node_size > utils::context().node_rank) ? 1 : 0;
      size_t NQ_offset = NQ_local * utils::context().node_rank +
                         ((_NQ % utils::context().node_size > utils::context().node_rank) ? 0 : (_NQ % utils::context().node_size));
      NQ_offset = (NQ_offset >= _NQ) ? 0 : NQ_offset;
      statistics.start("Direct");
      // Direct diagram
      MatrixXcd  X1(_nao, _nao);
      ztensor<3> v(NQ_local, _nao, _nao);
      ztensor<2> upper_Coul(_NQ, 1);
      MMatrixXcd X1m(X1.data(), _nao * _nao, 1);
      MMatrixXcd vm(v.data(), NQ_local, _nao * _nao);
      MMatrixXcd upper_Coul_m(upper_Coul.data() + NQ_offset, NQ_local, 1);
      if(NQ_local > 0) {
        // TODO: We can save a lot of time here by reducing the loop over ikp, but that would require removing the NQ_local, NQ_offset framework.
        //        The symmetry speedup only works if we load all of NQ at once -- which shold be doable now that we have much smaller memory footprint.
        for (int ikps = utils::context().internode_rank; ikps < _nk * _ns; ikps += utils::context().internode_size) {
          int is    = ikps % _ns;
          int ikp   = ikps / _ns;
          statistics.start("Read Coulomb Up");
          coul_int1.read_integrals(ikp, ikp);
          statistics.end();

          coul_int1.symmetrize(v, ikp, ikp, NQ_offset, NQ_local);

          X1 = _bz_utils.k_symmetry().value_AO(dm(is), ikp).transpose();
          // (Q, 1) = (Q, ab) * (ab, 1)
          upper_Coul_m += vm * X1m;
        }
      }
      statistics.start("Reduce Direct");
      MPI_Allreduce(MPI_IN_PLACE, upper_Coul.data(), upper_Coul.size(), MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, utils::context().global);
      statistics.end();

      upper_Coul /= double(_nk);
      if (NQ_local > 0) {
        for (int ii = utils::context().internode_rank; ii < _ink * _ns; ii += utils::context().internode_size) {
          int is   = ii / _ink;
          int ik   = ii % _ink;
          int k_ir = _bz_utils.k_symmetry().full_point(ik);
          statistics.start("Read Coulomb Low");
          coul_int1.read_integrals(k_ir, k_ir);
          statistics.end();
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
      MatrixXcd dmm(_nao, _nao);
      double     prefactor = (_ns == 2) ? 1.0 : 0.5;
      statistics.start("Exchange");
      if (NQ_local > 0) {
        for (int ii = utils::context().internode_rank; ii < _ink * _ns; ii += utils::context().internode_size) {
          int        is   = ii / _ink;
          int        ik   = ii % _ink;
          int        k_ir = _bz_utils.k_symmetry().full_point(ik);
          MMatrixXcd Fmm(new_Fock.data() + is * _ink * _nao * _nao + ik * _nao * _nao, _nao, _nao);
          for (int ikp = 0; ikp < _nk; ++ikp) {
            dmm = _bz_utils.k_symmetry().value_AO(dm(is), ikp);
            statistics.start("Read Coulomb Exch");
            coul_int1.read_integrals(k_ir, ikp);
            statistics.end();
            // (Q, i, b) or conj(Q, j, a)
            coul_int1.symmetrize(v, k_ir, ikp, NQ_offset, NQ_local);

            // (Qi, a) = (Qi, b) * (b, a)
            Ym = vmm * dmm;

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
      for (int ii = utils::context().global_rank; ii < _ns * _ink; ii += utils::context().global_size) {
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
    utils::allreduce(MPI_IN_PLACE, new_Fock.data(), new_Fock.size(), MPI_C_DOUBLE_COMPLEX, MPI_SUM, utils::context().global);
    statistics.end();
    statistics.end();
    statistics.print(utils::context().global);
    return new_Fock;
  }

  ztensor<4> hf_x2c_cpu_kernel::solve(const ztensor<4>& dm) {
    statistics.start("X2C Hartree-Fock");
    ztensor<4> new_Fock(1, _ink, _nso, _nso);
    new_Fock.set_zero();
    {
      df_integral_t coul_int1(_hf_path, _nao, _NQ, _bz_utils);

      size_t NQ_local = _NQ / utils::context().node_size;
      NQ_local += (_NQ % utils::context().node_size > utils::context().node_rank) ? 1 : 0;
      size_t NQ_offset = NQ_local * utils::context().node_rank +
                         ((_NQ % utils::context().node_size > utils::context().node_rank) ? 0 : (_NQ % utils::context().node_size));
      NQ_offset = (NQ_offset >= _NQ) ? 0 : NQ_offset;
      ztensor<3> v(NQ_local, _nao, _nao);
      // Direct diagram
      statistics.start("X2C direct diagram");
      {
        MatrixXcd  X1(_nao, _nao);
        MMatrixXcd X1m(X1.data(), _nao * _nao, 1);

        MMatrixXcd vm(v.data(), NQ_local, _nao * _nao);

        ztensor<2> upper_Coul(_NQ, 1);
        upper_Coul.set_zero();
        MMatrixXcd upper_Coul_m(upper_Coul.data() + NQ_offset, NQ_local, 1);
        for (int ikp = utils::context().internode_rank; ikp < _nk; ikp += utils::context().internode_size) {
          coul_int1.read_integrals(ikp, ikp);
          if(NQ_local > 0) {
            coul_int1.symmetrize(v, ikp, ikp, NQ_offset, NQ_local);

            // Sum of alpha-alpha and beta-beta spin block
            // In the presence of TR symmetry,
            // (dm_aa(-k) + dm_bb(-k)) = (dm_aa(k) + dm_bb(k))*
            MatrixXcd dm_so = _bz_utils.k_symmetry().value_AO(dm(0), ikp);
            X1 = dm_so.block(0, 0, _nao, _nao) + dm_so.block(_nao, _nao, _nao, _nao);
            X1 = X1.transpose().eval();
            // (Q, 1) = (Q, ab) * (ab, 1)
            upper_Coul_m += vm * X1m;
          }
        }
        statistics.start("Reduce Direct");
        MPI_Allreduce(MPI_IN_PLACE, upper_Coul.data(), upper_Coul.size(), MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, utils::context().global);
        statistics.end();
        upper_Coul /= double(_nk);

        MatrixXcd  Fm(1, _nao * _nao);
        MMatrixXcd Fmm(Fm.data(), _nao, _nao);
        for (int ik = utils::context().internode_rank; ik < _ink; ik += utils::context().internode_size) {
          int k_ir = _bz_utils.k_symmetry().full_point(ik);

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

      // One iteration per IBZ k-point: compute aa, bb, and ab Fock contributions
      // in a single ikp pass so that value_AO, read_integrals, and symmetrize are
      // each called once per ikp instead of once per (spin block × ikp).
      // The Madelung uses the AO overlap S_aa (= S_bb) on both sides for all blocks —
      // this is S_AO (spin-independent), not the spinor off-diagonal S_ab which is zero.
      for (int ik = utils::context().internode_rank; ik < (int)_ink; ik += utils::context().internode_size) {
        int         k_ir  = _bz_utils.k_symmetry().full_point(ik);
        MMatrixXcd  Fm_nso(new_Fock.data() + ik * _nso * _nso, _nso, _nso);
        CMMatrixXcd Sm_nso(_S_k.data() + ik * _nso * _nso, _nso, _nso);
        CMMatrixXcd dm_nso(dm.data() + ik * _nso * _nso, _nso, _nso);
        MatrixXcd   S_aa   = Sm_nso.block(0, 0, _nao, _nao);
        MatrixXcd   Fock_aa = MatrixXcd::Zero(_nao, _nao);
        MatrixXcd   Fock_bb = MatrixXcd::Zero(_nao, _nao);
        MatrixXcd   Fock_ab = MatrixXcd::Zero(_nao, _nao);

        for (int ikp = 0; ikp < _nk; ++ikp) {
          coul_int1.read_integrals(k_ir, ikp);
          if (NQ_local > 0) {
            coul_int1.symmetrize(v, k_ir, ikp, NQ_offset, NQ_local);
            // value_AO called once per ikp; all three spin blocks extracted from result.
            MatrixXcd dm_k = _bz_utils.k_symmetry().value_AO(dm(0), ikp);
            v2m = vmm.conjugate().transpose();  // same for all spin blocks at this ikp
            // alpha-alpha
            Ym = vmm * dm_k.block(0, 0, _nao, _nao); Y1m = Ymm.transpose();
            Fock_aa -= Y1mm * v2mm / double(_nk);
            // beta-beta
            Ym = vmm * dm_k.block(_nao, _nao, _nao, _nao); Y1m = Ymm.transpose();
            Fock_bb -= Y1mm * v2mm / double(_nk);
            // alpha-beta
            Ym = vmm * dm_k.block(0, _nao, _nao, _nao); Y1m = Ymm.transpose();
            Fock_ab -= Y1mm * v2mm / double(_nk);
          }
        }

        // Apply exchange + Madelung to aa and bb diagonal blocks.
        Fm_nso.block(0,    0,    _nao, _nao) += Fock_aa - _madelung * S_aa * dm_nso.block(0,    0,    _nao, _nao).eval() * S_aa;
        Fm_nso.block(_nao, _nao, _nao, _nao) += Fock_bb - _madelung * S_aa * dm_nso.block(_nao, _nao, _nao, _nao).eval() * S_aa;
        // Apply to ab block; ba is its adjoint.
        Fock_ab -= _madelung * S_aa * dm_nso.block(0, _nao, _nao, _nao).eval() * S_aa;
        Fm_nso.block(0,    _nao, _nao, _nao) += Fock_ab;
        Fm_nso.block(_nao, 0,    _nao, _nao)  = Fm_nso.block(0, _nao, _nao, _nao).transpose().conjugate();
      }
      statistics.end();
    }

    statistics.start("Reduce Fock");
    utils::allreduce(MPI_IN_PLACE, new_Fock.data(), new_Fock.size(), MPI_C_DOUBLE_COMPLEX, MPI_SUM, utils::context().global);
    statistics.end();
    statistics.end();
    statistics.print(utils::context().global);
    return new_Fock;
  }

}
