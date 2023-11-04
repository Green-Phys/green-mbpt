/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#include "green/mbpt/gw_solver.h"

#include <unsupported/Eigen/MatrixFunctions>

#include "green/mbpt/common_utils.h"

namespace green::mbpt {

  void gw_solver::solve(G_type& g, S1_type& sigma1, St_type& sigma_tau) {
    _coul_int1                 = new df_integral_t(_path, _nao, _NQ, _bz_utils);
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
    delete _coul_int1;
  }

  void gw_solver::setup_subcommunicator(size_t& kbatch_size, size_t& num_kbatch) {
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

  void gw_solver::setup_subcommunicator2(int q) {
    MPI_Comm_split(utils::context.global, q, utils::context.global_rank,
                   &_tau_comm2);  // Build subcommunicator over tau only not spin
    MPI_Comm_rank(_tau_comm2, &_tauid);
    MPI_Comm_size(_tau_comm2, &_ntauprocs);
    _spinid     = 0;
    _nspinprocs = 1;
    if (!utils::context.global_rank) std::cout << "ntauprocs = " << _ntauprocs << std::endl;
  }

  void gw_solver::selfenergy_innerloop(size_t q_ir, MPI_Comm subcomm, const G_type& G, St_type& Sigma) {
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
    if (!_second_only) {
      // Solve Dyson-like eqn of P(iOmega_{n}) through Chebyshev convolution
      eval_P_tilde(q_ir);
    }
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

  void gw_solver::read_next(const std::array<size_t, 4>& k) {
    // k = (k1, 0, q, k1+q) or (k1, q, 0, k1-q)
    size_t k1  = k[0];
    size_t k1q = k[3];
    _coul_int1->read_integrals(k1, k1q);
  }

  void gw_solver::symmetrize_P0() {
    size_t tau_batch = (_nts / 2) / (_ntauprocs);
    size_t tau_rest  = (_nts / 2) % (_ntauprocs);
    size_t t_start   = (_tauid < tau_rest) ? _tauid * (tau_batch + 1) : _tauid * tau_batch + tau_rest;
    size_t t_end     = (_tauid < tau_rest) ? (_tauid + 1) * (tau_batch + 1) : (_tauid + 1) * tau_batch + tau_rest;
    for (size_t it = t_start; it < t_end; ++it) {  // Symmetry of tau: P0(t) = P0(beta - t)
      matrix(_P0_tilde(_nts - it - 1, 0)) = matrix(_P0_tilde(it, 0));
    }
    // Hermitization
    hermitization(_P0_tilde);
  }

  void gw_solver::eval_P_tilde(const int q_ir) {
    // Matsubara P0_tilde_b
    ztensor<4> P0_w(_nw_b, 1, _NQ, _NQ);

    eval_P_tilde_w(q_ir, _P0_tilde, P0_w);
  }

  void gw_solver::eval_P_tilde_w(int q_ir, ztensor<4>& P0_tilde, ztensor<4>& P0_w) {
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
    // Transform back to intermediate Chebyshev representation for P_tilde
    if (_tauid == 0 and _spinid == 0 and q_ir == 0) {
      print_leakage(_ft.check_chebyshev(_P0_tilde), "P");
    }
  }

}  // namespace green::mbpt
