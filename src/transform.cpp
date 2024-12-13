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

#include "green/transform/transform.h"

#include <green/mbpt/common_defs.h>
#include <math.h>

#include <filesystem>
#include <iostream>

namespace green::transform {
  int int_transformer::find_pos(const tensor<double, 1>& k, const tensor<double, 2>& kmesh) {
    for (int i = 0; i < kmesh.shape()[0]; ++i) {
      bool found = true;
      for (int j = 0; j < k.shape()[0]; ++j) {
        found &= std::abs(k(j) - kmesh(i, j)) < 1e-12;
      }
      if (found) {
        return i;
      }
    }
    throw std::logic_error("K point (" + std::to_string(k(0)) + ", " + std::to_string(k(1)) + ", " + std::to_string(k(2)) +
                           ") has not been found in the mesh.");
  }

  tensor<double, 1> int_transformer::wrap(const tensor<double, 1>& k) {
    tensor<double, 1> kk = k;
    for (int j = 0; j < kk.shape()[0]; ++j) {
      while ((kk(j) - 9.9999999999e-1) > 0.0) {
        kk(j) -= 1.0;
      }
      if (std::abs(kk(j)) < 1e-9) {
        kk(j) = 0.0;
      }
      while (kk(j) < 0) {
        kk(j) += 1.0;
      }
    }
    return kk;
  };

  void int_transformer::get_mom_cons(h5pp::archive& file, int nk) {
    tensor<double, 2> kmesh(nk, 3);
    file["grid/k_mesh_scaled"] >> kmesh;

    tensor<double, 2> qmesh(kmesh.shape());
    for (int j = 0; j < nk; ++j) {
      dtensor<1> ki = kmesh(0);
      dtensor<1> kj = kmesh(j);
      auto       kq = wrap(ki - kj);
      for (int i = 0; i < kq.shape()[0]; ++i) {
        qmesh(j, i) = kq(i);
      }
    }

    for (int i = 0; i < nk; ++i) {
      dtensor<1> ki = kmesh(i);
      for (int j = 0; j < nk; ++j) {
        dtensor<1> kj = kmesh(j);
        auto       kq = wrap(ki - kj);
        int        q  = find_pos(kq, qmesh);
        _q_ind(j, q)  = i;
        _q_ind2(i, j) = q;
      }
    }
  }

  int int_transformer::mom_cons(int i, int j, int k) const {
    int q = _q_ind2(i, j);
    int l = _q_ind(k, q);
    return l;
  }

  void int_transformer::read_integrals(h5pp::archive& file, int current_chunk, int chunk_size, ztensor<4>& vij_Q) {
    // Integral dataset key
    std::string inner   = "/" + std::to_string(current_chunk * chunk_size);
    std::string dsetnum = "VQ" + inner;
    file[dsetnum] >> vij_Q;
  }

  int int_transformer::irre_pos_kpair(int idx, std::vector<int>& kpair_irre_list) {
    std::vector<int>::iterator itr   = std::find(kpair_irre_list.begin(), kpair_irre_list.end(), idx);
    int                        index = std::distance(kpair_irre_list.begin(), itr);
    return index;
  }

  void int_transformer::get_ki_kj(const int kikj, int& ki, int& kj, int nkpts) {
    for (int i = 1; i <= nkpts; ++i) {
      if (kikj < i * (i + 1) / 2) {
        ki = i - 1;
        break;
      }
    }
    kj = kikj - ki * (ki + 1) / 2;
  }

  int int_transformer::get_idx_red(int k1, int k2, std::vector<int>& conj_kpair_list, std::vector<int>& trans_kpair_list,
                                   std::vector<int>& kpair_irre_list) {
    int idx = (k1 >= k2) ? k1 * (k1 + 1) / 2 + k2 : k2 * (k2 + 1) / 2 + k1;  // k-pair = (k1, k2) or (k2, k1)
    // Corresponding symmetry-related k-pair
    if (conj_kpair_list[idx] != idx) {
      idx = conj_kpair_list[idx];
    } else if (trans_kpair_list[idx] != idx) {
      idx = trans_kpair_list[idx];
    }
    int idx_red = irre_pos_kpair(idx, kpair_irre_list);
    return idx_red;
  }

  int int_transformer::get_vtype(int k1, int k2, std::vector<int>& conj_kpair_list, std::vector<int>& trans_kpair_list) {
    int idx  = (k1 >= k2) ? k1 * (k1 + 1) / 2 + k2 : k2 * (k2 + 1) / 2 + k1;  // k-pair = (k1, k2) or (k2, k1)
    int sign = (k1 >= k2) ? 1 : -1;
    // determine type
    if (conj_kpair_list[idx] != idx) {
      return 2 * sign;
    } else if (trans_kpair_list[idx] != idx) {
      return 3 * sign;
    }
    return 1 * sign;
  }

  void int_transformer::symmetrization(ztensor<3>& v, int v_type) {
    int NQ  = v.shape()[0];
    int nno = v.shape()[1];
    if (v_type < 0) {
      for (int Q = 0; Q < NQ; ++Q) {
        MMatrixXcd v_m(v.data() + Q * nno * nno, nno, nno);
        v_m = v_m.conjugate().transpose().eval();
      }
    }
    if (std::abs(v_type) == 2) {  // conjugate
      for (int Q = 0; Q < NQ; ++Q) {
        MMatrixXcd v_m(v.data() + Q * nno * nno, nno, nno);
        v_m = v_m.conjugate();
      }
    } else if (std::abs(v_type) == 3) {  // transpose
      for (int Q = 0; Q < NQ; ++Q) {
        MMatrixXcd v_m(v.data() + Q * nno * nno, nno, nno);
        v_m = v_m.transpose().eval();
      }
    }
  }

  void int_transformer::transform_3point() {
    int myid, nprocs;

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    std::string basename = _params.in_int_file;

    int         NQ       = 0;
    int         nao      = 0;
    if (!myid) {
      std::string   V0 = basename + "/VQ_0.h5";
      h5pp::archive v0_file(V0, "r");
      dtensor<4>    buffer_in_d;
      v0_file[std::to_string(0)] >> buffer_in_d;
      ztensor<4> buffer_in = buffer_in_d.view<std::complex<double>>();
      NQ                   = buffer_in.shape()[1];
      nao                  = buffer_in.shape()[2];
      v0_file.close();
    }
    MPI_Bcast(&NQ, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nao, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Read impurity information
    int                     nimp = 0;
    ztensor<3>              X_k;
    std::vector<dtensor<2>> UUs;
    {
      h5pp::archive tr_file(_params.input_file, "r");
      tr_file["nimp"] >> nimp;
      std::string trans = "X_k";
      tr_file[trans] >> X_k;
      for (int i = 0; i < nimp; ++i) {
        dtensor<2> uu;
        trans = std::to_string(i) + "/UU";
        tr_file[trans] >> uu;
        UUs.push_back(uu);
      }
      tr_file.close();
    }

    h5pp::archive out_file;
    if (myid == 0) out_file.open(_params.out_int_file, "a");
    // For gtest
    std::vector<std::complex<double>> zERI_imp1;

    for (int i = 0; i < nimp; ++i) {
      if (!myid) std::cout << "Process impurity " << i << std::endl;
      dtensor<2>& UU  = UUs[i];
      int         nno = UU.shape()[0];

      orthogonalize_Vij_Q(myid, basename, X_k, UU, i, nao, nno, NQ);
      auto       int_file = h5pp::archive("reduced.int." + std::to_string(i) + ".h5", "r");
      ztensor<4> VijQ1(_chunk_size, NQ, nno, nno);
      ztensor<4> VijQ2(_chunk_size, NQ, nno, nno);

      // Compute local VijQ (we don't use this in practice)
      ztensor<3> VijQ_loc(VijQ1.shape()[1], VijQ1.shape()[2], VijQ1.shape()[3]);
      compute_local_VijQ(myid, nprocs, NQ, nno, int_file, VijQ1, VijQ_loc);

      // Compute local Vijkl
      dtensor<4> dERI(nno, nno, nno, nno);
      ztensor<4> zERI(nno, nno, nno, nno);
      extract_impurity_interaction(myid, nprocs, int_file, VijQ1, VijQ2, dERI, zERI, nno, NQ);

      // Compute decomposed local VijQ_imp
      auto VijQ_imp = decomp_interaction(dERI);

      if (myid == 0) {
        out_file[std::to_string(i) + "/interaction"] << dERI;
        int         chunkid  = 0;
        std::string dir_name = _params.dc_path + std::to_string(nimp);
        std::filesystem::create_directory(dir_name);
        std::string   fname = dir_name + "/VQ_0.h5";
        h5pp::archive ar(fname, "w");
        ar["/" + std::to_string(chunkid)] << VijQ_imp;
        ar.close();

        int           nq        = VijQ_imp.shape()[2];
        int           chunksize = nao * nao * nq * 16;
        std::string   metaname  = dir_name + "/meta.h5";
        hid_t         metafile  = H5Fopen(metaname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        h5pp::archive meta(metaname);
        meta["/chunk_indices"] << chunkid;
        meta["/chunk_size"] << chunksize;
        ar.close();
      }
    }
    if (myid == 0) {
      out_file.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  void int_transformer::extract_impurity_interaction(int myid, int nprocs, h5pp::archive& int_file, ztensor<4>& VijQ1,
                                                     ztensor<4>& VijQ2, dtensor<4>& dERI, ztensor<4>& zERI, int nno, int NQ) {
    MMatrixXcd zERI_m(zERI.data(), nno * nno, nno * nno);
    MMatrixXd  dERI_m(dERI.data(), nno * nno, nno * nno);
    int        current_chunk1 = -1;
    int        current_chunk2 = -1;
    if (myid == 0) std::cout << "Compute local Vijkl ..." << std::endl;
    for (int k1k2 = myid; k1k2 < _nkpts * _nkpts; k1k2 += nprocs) {
      int k1 = k1k2 / _nkpts;
      int k2 = k1k2 % _nkpts;
      std::cout << k1k2 + 1 << " of " << _nkpts * _nkpts << std::endl;

      int idx_red    = get_idx_red(k1, k2, _conj_kpair_list, _trans_kpair_list, _kpair_irre_list);
      int v_type     = get_vtype(k1, k2, _conj_kpair_list, _trans_kpair_list);
      current_chunk1 = idx_red / _chunk_size;
      int idx_wrap   = idx_red % _chunk_size;
      read_integrals(int_file, current_chunk1, _chunk_size, VijQ1);
      ztensor<3> v = VijQ1(idx_wrap);
      symmetrization(v, v_type);
      CMMatrixXcd vc_1(v.data(), NQ, nno * nno);

      for (int k3 = 0; k3 < _nkpts; ++k3) {
        int k4         = mom_cons(k1, k2, k3);

        int idx_red_2  = get_idx_red(k3, k4, _conj_kpair_list, _trans_kpair_list, _kpair_irre_list);
        int v_type_2   = get_vtype(k3, k4, _conj_kpair_list, _trans_kpair_list);
        current_chunk2 = idx_red_2 / _chunk_size;
        int idx_wrap_2 = idx_red_2 % _chunk_size;
        read_integrals(int_file, current_chunk2, _chunk_size, VijQ2);
        ztensor<3> v2 = VijQ2(idx_wrap_2);
        symmetrization(v2, v_type_2);
        CMMatrixXcd vc_2(v2.data(), NQ, nno * nno);

        zERI_m += vc_1.transpose() * vc_2 / (_nkpts * _nkpts * _nkpts);
      }
    }

    MPI_Allreduce(MPI_IN_PLACE, zERI.data(), zERI.size(), MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);

    dERI_m = zERI_m.real();
  }

  void int_transformer::compute_local_VijQ(int myid, int nprocs, int NQ, int nno, h5pp::archive& int_file, ztensor<4>& VijQ1,
                                           ztensor<3>& VijQ_loc) {
    int current_chunk1 = -1;
    VijQ_loc.set_zero();
    for (int k1 = myid; k1 < _nkpts; k1 += nprocs) {
      for (int k2 = 0; k2 < _nkpts; ++k2) {
        int idx_red    = get_idx_red(k1, k2, _conj_kpair_list, _trans_kpair_list, _kpair_irre_list);
        int v_type     = get_vtype(k1, k2, _conj_kpair_list, _trans_kpair_list);
        current_chunk1 = idx_red / _chunk_size;
        int idx_wrap   = idx_red % _chunk_size;
        read_integrals(int_file, current_chunk1, _chunk_size, VijQ1);
        ztensor<3> v = VijQ1(idx_wrap);
        symmetrization(v, v_type);
        CMMatrixXcd vc_1(v.data(), NQ, nno * nno);

        MMatrixXcd  vc_2(VijQ_loc.data(), NQ, nno * nno);
        vc_2 += vc_1;  // / (nkpts*nkpts);
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, VijQ_loc.data(), VijQ_loc.size(), MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
  }

  void int_transformer::orthogonalize_Vij_Q(int myid, const std::string& basename, ztensor<3>& X_k, dtensor<2>& UU, int i,
                                            int nao, int nno, int NQ) {
    MMatrixXd     UUM(UU.data(), nno, nao);
    h5pp::archive int_file;
    // Transform V^{kikj}_{ij}(Q) to orthogonal basis
    if (myid == 0 && _params.transform) {
      ztensor<4> buffer_in(_chunk_size, NQ, nao, nao);
      ztensor<4> buffer_out(_chunk_size, NQ, nno, nno);
      int_file.open("reduced.int." + std::to_string(i) + ".h5", "w");
      for (int ic = 0; ic < _nchunks; ++ic) {
        std::cout << "Chunk " << ic + 1 << " out of " << _nchunks << std::endl;
        int           flat_index   = ic * _chunk_size;
        std::string   name         = "VQ/" + std::to_string(flat_index);
        std::string   vq_file_path = basename + "/VQ_" + std::to_string(flat_index) + ".h5";
        h5pp::archive vq_file(vq_file_path, "r");
        vq_file[std::to_string(flat_index)] >> buffer_in.view<double>();
        vq_file.close();

        for (int ikk = 0; ikk < _chunk_size; ++ikk) {
          int iii = flat_index + ikk;
          if (iii >= _num_kpair_stored) {
            break;
          }
          int kikj = _kpair_irre_list[iii];  // k1*(k1+1)/2 + k2, k2 <= k1
          int ki   = -1;
          int kj   = -1;
          get_ki_kj(kikj, ki, kj, _nkpts);

          MMatrixXcd X_i_M(X_k.data() + ki * X_k.shape()[1] * X_k.shape()[2], X_k.shape()[1], X_k.shape()[2]);
          MMatrixXcd X_j_M(X_k.data() + kj * X_k.shape()[1] * X_k.shape()[2], X_k.shape()[1], X_k.shape()[2]);
          // Orthogoanlization and projection to active space
          for (int iQ = 0; iQ < NQ; ++iQ) {
            MMatrixXcd in(buffer_in.data() + ikk * NQ * nao * nao + iQ * nao * nao, nao, nao);
            MMatrixXcd out(buffer_out.data() + ikk * NQ * nno * nno + iQ * nno * nno, nno, nno);
            out = (UUM * X_i_M.adjoint()) * in * (X_j_M * UUM.transpose());
          }
        }
        int_file[name] << buffer_out;
      }
      int_file.close();
      std::cout << "Finish orthogonalization and projection for Vij_Q" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  dtensor<3> decomp_interaction(const dtensor<4>& dERI, double atol) {
    int nao = dERI.shape()[0];

    for (int i = 1; i < 4; i++) {
      if (dERI.shape()[i] != (size_t)nao) throw std::runtime_error("[decomp] full tensor dimension mismatch");
    }

    int        nao2 = nao * nao;
    CMMatrixXd Umap(dERI.data(), nao2, nao2);

    if ((Umap - Umap.transpose()).norm() > 1e-12) throw std::runtime_error("[decomp] full tensor is not symmetric");

    Eigen::SelfAdjointEigenSolver<MatrixXd> eigen_solver(Umap);
    if (eigen_solver.info() != Eigen::Success) throw std::runtime_error("[decomp] eigen decomposition failed");

    int ncut = 0;
    for (int i = 0; i < nao2; i++) {
      double val = eigen_solver.eigenvalues()(i);
      if (std::abs(val) >= atol) {
        if (val < 0)
          throw std::runtime_error("[decomp] negative eigenvalue " + std::to_string(val) + " encountered with tolarance " +
                                   std::to_string(val));
        break;
      }
      ncut++;
    }

    int        nq = nao2 - ncut;

    dtensor<3> VijQ_imp(nao, nao, nq);

    for (int i = 0; i < nao; i++) {
      for (int j = 0; j < nao; j++) {
        for (int alpha = 0; alpha < nq; alpha++) {
          VijQ_imp(i, j, alpha) =
              std::sqrt(eigen_solver.eigenvalues()(alpha + ncut)) * eigen_solver.eigenvectors()(i * nao + j, alpha + ncut);
        }
      }
    }
    return VijQ_imp;
  }
};  // namespace green::transform
// namespace green::transform
