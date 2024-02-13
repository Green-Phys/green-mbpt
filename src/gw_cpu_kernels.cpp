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
#include <green/mbpt/df_integral_t.h>
#include <green/mbpt/kernels.h>
#include <green/utils/mpi_shared.h>
#include <green/utils/mpi_utils.h>

namespace green::mbpt::kernels {

  void gw_kernel::solve(G_type& g, St_type& sigma_tau) {
    _coul_int1                 = new df_integral_t(_path, _nao, _NQ, _bz_utils);
    _P0_tilde.resize(_nts, 1, _NQ, _NQ);
    MPI_Datatype dt_matrix     = utils::create_matrix_datatype<std::complex<double>>(_nso * _nso);
    MPI_Op       matrix_sum_op = utils::create_matrix_operation<std::complex<double>>();
    auto&        sigma_fermi   = sigma_tau.object();
    sigma_tau.fence();
    if (!utils::context.node_rank) sigma_fermi.set_zero();
    sigma_tau.fence();
    // Deal with k-batches where each k-batch has (nprocs/ntauspin_mpi) k-points or (ink) k-points if ntau_mpi=1.
    size_t kbatch_size;
    size_t num_kbatch;
    statistics.start("total");
    statistics.start("kbatches");
    setup_subcommunicator(kbatch_size, num_kbatch);
    sigma_tau.fence();
    for (size_t batch_id = 0; batch_id < num_kbatch; ++batch_id) {
      size_t q    = utils::context.global_rank / _ntauspin_mpi + batch_id * kbatch_size;
      size_t q_ir = _bz_utils.symmetry().full_point(q);
      selfenergy_innerloop(q_ir, _tauspin_comm, g, sigma_tau);
    }
    sigma_tau.fence();
    statistics.end();
    statistics.start("krest");
    // Process the rest k points: (_ink%kbatch_size) k points
    size_t numk_rest = _ink % (kbatch_size);
    sigma_tau.fence();
    if (numk_rest != 0) {
      if (!utils::context.global_rank) std::cout << "###############################" << std::endl;
      if (!utils::context.global_rank) std::cout << "Process the rest k-points" << std::endl;
      if (!utils::context.global_rank) std::cout << "###############################" << std::endl;
      size_t q_id = utils::context.global_rank % numk_rest;
      size_t q    = q_id + num_kbatch * kbatch_size;
      size_t q_ir = _bz_utils.symmetry().full_point(q);
      setup_subcommunicator2(q_ir);
      selfenergy_innerloop(q_ir, _tau_comm2, g, sigma_tau);
    }
    sigma_tau.fence();
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
    if (_q0_utils.q0_treatment() == extrapolate) {
      MPI_Allreduce(MPI_IN_PLACE, _eps_inv_wq.data(), _eps_inv_wq.size(), MPI_C_DOUBLE_COMPLEX, MPI_SUM, utils::context.global);
      _q0_utils.GW_q0_correction(_eps_inv_wq, sigma_fermi, g.object(), _ft, _X2C, utils::context.global_rank,
                                 utils::context.node_rank, utils::context.node_size, sigma_tau.win());
    }
    statistics.end();
    statistics.print(utils::context.global);
    if (numk_rest != 0) {
      MPI_Comm_free(&_tau_comm2);
    }
    if (_ntauspin_mpi > 1) {
      MPI_Comm_free(&_tauspin_comm);
    }
    MPI_Type_free(&dt_matrix);
    MPI_Op_free(&matrix_sum_op);
    _P0_tilde.resize(0, 0, 0, 0);
    delete _coul_int1;
  }

  void gw_kernel::setup_subcommunicator(size_t& kbatch_size, size_t& num_kbatch) {
    if (!utils::context.global_rank) std::cout << "Number of processes = " << utils::context.global_size << std::endl;
    if (_ntauspin_mpi > 1) {
      int ntauspinprocs;
      int tauspinid;
      MPI_Comm_split(utils::context.global, utils::context.global_rank / _ntauspin_mpi, utils::context.global_rank,
                     &_tauspin_comm);  // Build subcommunicator
      MPI_Comm_rank(_tauspin_comm, &tauspinid);
      MPI_Comm_size(_tauspin_comm, &ntauspinprocs);
      assert(tauspinid == utils::context.global_rank % _ntauspin_mpi);
      assert(ntauspinprocs == _ntauspin_mpi);
      _tauid      = tauspinid / _ns;
      _spinid     = tauspinid % _ns;
      _nspinprocs = _ns;
      _ntauprocs  = _ntauspin_mpi / _ns;
    } else {
      _tauid      = 0;
      _spinid     = 0;
      _nspinprocs = 1;
      _ntauprocs  = 1;
    }
    kbatch_size = utils::context.global_size / _ntauspin_mpi;
    num_kbatch  = _ink / kbatch_size;
    if (!utils::context.global_rank) std::cout << "kbatch_size = " << kbatch_size << std::endl;
    if (!utils::context.global_rank) std::cout << "num_kbatch = " << num_kbatch << std::endl;
    if (!utils::context.global_rank) std::cout << "ntauprocs = " << _ntauprocs << std::endl;
    if (!utils::context.global_rank) std::cout << "nspinprocs = " << _nspinprocs << std::endl;
  }

  void gw_kernel::setup_subcommunicator2(int q) {
    MPI_Comm_split(utils::context.global, q, utils::context.global_rank,
                   &_tau_comm2);  // Build subcommunicator over tau only not spin
    MPI_Comm_rank(_tau_comm2, &_tauid);
    MPI_Comm_size(_tau_comm2, &_ntauprocs);
    _spinid     = 0;
    _nspinprocs = 1;
    if (!utils::context.global_rank) std::cout << "ntauprocs = " << _ntauprocs << std::endl;
  }

  void gw_kernel::selfenergy_innerloop(size_t q_ir, MPI_Comm subcomm, const G_type& G, St_type& Sigma) {
    _P0_tilde.set_zero();
    for (size_t k1 = 0; k1 < _nk; ++k1) {
      std::array<size_t, 4> k = _bz_utils.momentum_conservation({
          {k1, 0, q_ir}
      });
      statistics.start("read");
      read_next(k);
      statistics.end();
      statistics.start("eval_P0_tilde");
      if (_p_sp) {  // Single-precision run
        eval_P0_tilde<std::complex<float>>(k, G);
      } else {  // Double-precision run
        eval_P0_tilde<std::complex<double>>(k, G);
      }
      statistics.end();
    }
    symmetrize_P0();
    statistics.start("P0_reduce");  // Reduction on tau axis
    if (_ntauprocs * _nspinprocs > 1)
      utils::allreduce(MPI_IN_PLACE, _P0_tilde.data(), _P0_tilde.size(), MPI_C_DOUBLE_COMPLEX, MPI_SUM, subcomm);
    statistics.end();
    _P0_tilde /= (_nk);
    statistics.start("eval_P_tilde");
    // Solve Dyson-like eqn of P(iOmega_{n}) through Chebyshev convolution
    eval_P_tilde(q_ir);
    statistics.end();
    for (size_t k1 = 0; k1 < _ink; ++k1) {
      size_t k1_ir = _bz_utils.symmetry().full_point(k1);
      // Loop over the degenerate points of q_ir
      for (size_t q_deg : _bz_utils.symmetry().deg(_bz_utils.symmetry().reduced_point(q_ir))) {
        std::array<size_t, 4> k = _bz_utils.momentum_conservation({
            {k1_ir, q_deg, 0}
        });
        statistics.start("read");
        read_next(k);
        statistics.end();
        statistics.start("eval_S");
        if (_sigma_sp) {
          eval_selfenergy<std::complex<float>>(k, G, Sigma);
        } else {
          eval_selfenergy<std::complex<double>>(k, G, Sigma);
        }
        statistics.end();
      }
    }
  }

  void gw_kernel::read_next(const std::array<size_t, 4>& k) {
    // k = (k1, 0, q, k1+q) or (k1, q, 0, k1-q)
    size_t k1  = k[0];
    size_t k1q = k[3];
    _coul_int1->read_integrals(k1, k1q);
  }

  void gw_kernel::symmetrize_P0() {
    size_t tau_batch = (_nts / 2) / (_ntauprocs);
    size_t tau_rest  = (_nts / 2) % (_ntauprocs);
    size_t t_start   = (_tauid < tau_rest) ? _tauid * (tau_batch + 1) : _tauid * tau_batch + tau_rest;
    size_t t_end     = (_tauid < tau_rest) ? (_tauid + 1) * (tau_batch + 1) : (_tauid + 1) * tau_batch + tau_rest;
    for (size_t it = t_start; it < t_end; ++it) {  // Symmetry of tau: P0(t) = P0(beta - t)
      matrix(_P0_tilde(_nts - it - 1, 0)) = matrix(_P0_tilde(it, 0));
    }
    // Hermitization
    make_hermitian(_P0_tilde);
  }

  void gw_kernel::eval_P_tilde(const int q_ir) {
    // Matsubara P0_tilde_b
    ztensor<4> P0_w(_nw_b, 1, _NQ, _NQ);

    eval_P_tilde_w(q_ir, _P0_tilde, P0_w);
  }

  void gw_kernel::eval_P_tilde_w(int q_ir, ztensor<4>& P0_tilde, ztensor<4>& P0_w) {
    // Transform P0_tilde from Fermionic tau to Bonsonic Matsubara grid
    _ft.tau_f_to_w_b(_P0_tilde, P0_w);

    if (_tauid == 0 and _spinid == 0 and q_ir == 0) {
      print_leakage(_ft.check_chebyshev(_P0_tilde), "P0");
    }

    // Solve Dyson-like eqn for ncheb frequency points
    MatrixXcd              identity = MatrixXcd::Identity(_NQ, _NQ);
    // Eigen::FullPivLU<MatrixXcd> lusolver(_NQ,_NQ);
    Eigen::LDLT<MatrixXcd> ldltsolver(_NQ);
    for (size_t n = 0; n < _nw_b; ++n) {
      MatrixXcd temp     = identity - matrix(P0_w(n, 0));
      // temp = lusolver.compute(temp).inverse().eval();
      temp               = ldltsolver.compute(temp).solve(identity).eval();
      temp               = 0.5 * (temp + temp.conjugate().transpose().eval());
      matrix(P0_w(n, 0)) = (temp * matrix(P0_w(n, 0))).eval();
    }

    // Transform back from Bosonic Matsubara to Fermionic tau.
    _ft.w_b_to_tau_f(P0_w, _P0_tilde);
    // for G0W0 correction
    if (_q0_utils.q0_treatment() == extrapolate and _tauid == 0 and _spinid == 0) {
      size_t iq = _bz_utils.symmetry().reduced_point(q_ir);
      _q0_utils.aux_to_PW_00(P0_w, _eps_inv_wq, iq);
    }
    // Transform back to intermediate Chebyshev representation for P_tilde
    if (_tauid == 0 and _spinid == 0 and q_ir == 0) {
      print_leakage(_ft.check_chebyshev(_P0_tilde), "P");
    }
  }

  template <typename prec>
  void gw_kernel::eval_P0_tilde(const std::array<size_t, 4>& k, const G_type& G) {
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
  void gw_kernel::assign_G(size_t k, size_t t, size_t s, const ztensor<5>& G_fermi, MatrixX<prec>& G_k) {
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
  void gw_kernel::assign_G_nso(size_t k, size_t t, size_t s1, size_t s2, const ztensor<5>& G_fermi, MatrixX<prec>& G_k) {
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
  void gw_kernel::P0_contraction(const MatrixX<prec>& Gb_k1, const MatrixX<prec>& G_k1q, MMatrixX<prec>& vm, MMatrixX<prec>& VVm,
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
  void gw_kernel::eval_selfenergy(const std::array<size_t, 4>& k, const G_type& G_fermi, St_type& Sigma_fermi_s) {
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
  void gw_kernel::selfenergy_contraction(const std::array<size_t, 4>& k, const MatrixX<prec>& G_k1q, MMatrixX<prec>& vm,
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
  template void gw_kernel::eval_P0_tilde<std::complex<float>>(const std::array<size_t, 4>& k, const G_type&);
  template void gw_kernel::eval_P0_tilde<std::complex<double>>(const std::array<size_t, 4>& k, const G_type&);
  template void gw_kernel::P0_contraction(const MatrixX<std::complex<float>>& Gb_k1, const MatrixX<std::complex<float>>& G_k1q,
                                          MMatrixX<std::complex<float>>& vm, MMatrixX<std::complex<float>>& VVm,
                                          MMatrixX<std::complex<float>>& VVmm, MMatrixX<std::complex<float>>& X1m,
                                          MMatrixX<std::complex<float>>& vmm, MMatrixX<std::complex<float>>& X2m,
                                          MMatrixX<std::complex<float>>& X1mm, MMatrixX<std::complex<float>>& X2mm,
                                          MMatrixXcd& P0, double& prefactor);
  template void gw_kernel::P0_contraction(const MatrixX<std::complex<double>>& Gb_k1, const MatrixX<std::complex<double>>& G_k1q,
                                          MMatrixX<std::complex<double>>& vm, MMatrixX<std::complex<double>>& VVm,
                                          MMatrixX<std::complex<double>>& VVmm, MMatrixX<std::complex<double>>& X1m,
                                          MMatrixX<std::complex<double>>& vmm, MMatrixX<std::complex<double>>& X2m,
                                          MMatrixX<std::complex<double>>& X1mm, MMatrixX<std::complex<double>>& X2mm,
                                          MMatrixXcd& P0, double& prefactor);
  template void gw_kernel::eval_selfenergy<std::complex<float>>(const std::array<size_t, 4>& k, const G_type&, St_type&);
  template void gw_kernel::eval_selfenergy<std::complex<double>>(const std::array<size_t, 4>& k, const G_type&, St_type&);
  template void gw_kernel::selfenergy_contraction(const std::array<size_t, 4>& k, const MatrixX<std::complex<float>>& G_k1q,
                                                  MMatrixX<std::complex<float>>& vm, MMatrixX<std::complex<float>>& Y1m,
                                                  MMatrixX<std::complex<float>>& Y1mm, MMatrixX<std::complex<float>>& Y2mm,
                                                  MMatrixX<std::complex<float>>& X2m, MMatrixX<std::complex<float>>& Y2mmm,
                                                  MMatrixX<std::complex<float>>& X2mm, MatrixX<std::complex<float>>& P,
                                                  MatrixXcd& Sm_ts);
  template void gw_kernel::selfenergy_contraction(const std::array<size_t, 4>& k, const MatrixX<std::complex<double>>& G_k1q,
                                                  MMatrixX<std::complex<double>>& vm, MMatrixX<std::complex<double>>& Y1m,
                                                  MMatrixX<std::complex<double>>& Y1mm, MMatrixX<std::complex<double>>& Y2mm,
                                                  MMatrixX<std::complex<double>>& X2m, MMatrixX<std::complex<double>>& Y2mmm,
                                                  MMatrixX<std::complex<double>>& X2mm, MatrixX<std::complex<double>>& P,
                                                  MatrixXcd& Sm_ts);
  template void gw_kernel::assign_G(size_t k, size_t t, size_t s, const ztensor<5>&, MatrixX<std::complex<double>>& G_k);
  template void gw_kernel::assign_G(size_t k, size_t t, size_t s, const ztensor<5>&, MatrixX<std::complex<float>>& G_k);
  template void gw_kernel::assign_G_nso(size_t k, size_t t, size_t s1, size_t s2, const ztensor<5>&,
                                        MatrixX<std::complex<double>>& G_k);
  template void gw_kernel::assign_G_nso(size_t k, size_t t, size_t s1, size_t s2, const ztensor<5>&,
                                        MatrixX<std::complex<float>>& G_k);

}