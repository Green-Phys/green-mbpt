/*
 * Copyright (c) 2021-2022 University of Michigan.
 *
 */

#include "green/mbpt/dyson.h"

#include <green/utils/timing.h>

#include "green/mbpt/common_utils.h"

namespace green::mbpt {

  template <typename G, typename S1, typename St>
  dyson<G, S1, St>::dyson(const params::params& p) :
      _ft(p), _bz_utils(p), _ncheb(_ft.sd().repn_fermi().ni()), _nts(_ft.sd().repn_fermi().nts()),
      _nw(_ft.sd().repn_fermi().nw()), _nk(_bz_utils.nk()), _ink(_bz_utils.ink()), _X2C(false), _mu(0.0),
      _const_density(p["const_density"]), _tol(p["tolerance"]) {
    dtensor<5>    S_k_tmp;
    dtensor<5>    H_k_tmp;
    h5pp::archive in_file(p["input_file"]);
    in_file["HF/S-k"] >> S_k_tmp;
    in_file["HF/H-k"] >> H_k_tmp;
    in_file["HF/Energy_nuc"] >> _E_nuc;
    in_file["params/ns"] >> _ns;
    in_file["params/nso"] >> _nso;
    in_file["params/nao"] >> _nao;
    in_file["params/nel_cell"] >> _nel;
    in_file.close();
    std::array<size_t, 4> shape;
    std::copy(S_k_tmp.shape().begin(), S_k_tmp.shape().end() - 1, shape.begin());
    _S_k.resize(_ns, _ink, _nso, _nso);
    _H_k.resize(_ns, _ink, _nso, _nso);
    for (size_t is = 0; is < _ns; ++is) {
      _S_k(is) << _bz_utils.full_to_ibz(S_k_tmp.view<std::complex<double>>().reshape(shape)(is));
      _H_k(is) << _bz_utils.full_to_ibz(H_k_tmp.view<std::complex<double>>().reshape(shape)(is));
    }
    make_hermitian(_S_k);
    make_hermitian(_H_k);
  }

  template <typename G, typename S1, typename St>
  void dyson<G, S1, St>::selfenergy_eigenspectra(const Sigma1& sigma1, const Sigma_tau& sigma_tau_s,
                                                 std::vector<std::complex<double>>& eigenvalues_Sigma_p_F) const {
    const ztensor<5>&                    sigma_tau = extract_local(sigma_tau_s);
    ztensor<2>                           X_k(_nso, _nso);
    ztensor<2>                           Sigma_w(_nso, _nso);
    size_t                               batch_number = 0;
    Eigen::ComplexEigenSolver<MatrixXcd> solver(_nso);
    // iterate over all matsubara frequencies
    for (size_t iwsk = utils::context.global_rank; iwsk < _nw * _ns * _ink; iwsk += utils::context.global_size) {
      size_t iw = iwsk / (_ns * _ink);
      size_t is = (iwsk % (_ns * _ink)) / _ink;
      size_t ik = iwsk % _ink;
      _ft.tau_to_omega_wsk(sigma_tau, Sigma_w, iw, is, ik, 1);
      X_k               = _H_k(is, ik) + sigma1(is, ik) + Sigma_w;
      MatrixXcd S_k_inv = matrix(_S_k(is, ik)).inverse().eval();
      column    ev      = solver.compute(S_k_inv * matrix(X_k), false).eigenvalues();
      // store eigenvalues for each frequency and k-point
      for (size_t i = 0; i < _nso; ++i) {
        eigenvalues_Sigma_p_F.push_back(ev(i, 0));
      }
    }
  }

  template <typename G, typename S1, typename St>
  double dyson<G, S1, St>::compute_number_of_electrons(double                                   mu,
                                                       const std::vector<std::complex<double>>& eigenvalues_Sigma_p_F) const {
    // Get density matrix for given mu
    MatrixXcd            TtBn = _ft.Ttn().block(_nts - 1, 0, 1, _nw);
    double               nel  = 0.0;
    MatrixXcd            trace_w(_nw, 1);
    MatrixXcd            trace_t(1, 1);
    std::complex<double> muomega;
    trace_w              = MatrixXcd::Zero(_nw, 1);
    size_t loc_loop_size = _nw * _ns * _ink / utils::context.global_size;
    loc_loop_size += (utils::context.global_rank < (_nw * _ns * _ink % utils::context.global_size)) ? 1 : 0;
    size_t inkk = (eigenvalues_Sigma_p_F.size()) / (loc_loop_size * _nso);
    for (size_t iwsk = utils::context.global_rank, iii = 0; iwsk < _nw * _ns * _ink; iwsk += utils::context.global_size) {
      size_t iw   = iwsk / (_ns * _ink);
      size_t is   = (iwsk % (_ns * _ink)) / _ink;
      size_t ik   = (iwsk % _ink);
      size_t k_ir = _bz_utils.symmetry().full_point(ik);
      muomega     = _ft.wsample_fermi()(iw) * 1.0i + mu;
      for (int ikk = 0; ikk < inkk; ++ikk) {
        // Trace over AO index
        for (size_t i = 0; i < _nso; ++i, ++iii) {
          trace_w(iw, 0) +=
              _bz_utils.nkpw() * _bz_utils.symmetry().weight()[k_ir] * (1.0 / inkk) / (muomega - eigenvalues_Sigma_p_F[iii]);
        }
      }
    }
    // Transform to tau
    trace_t       = TtBn * trace_w;
    int prefactor = (_ns == 2 or _X2C) ? 1 : 2;
    nel += prefactor * -trace_t(0, 0).real();
    MPI_Allreduce(MPI_IN_PLACE, &nel, 1, MPI_DOUBLE, MPI_SUM, utils::context.global);
    return nel;
  }

  template <typename G, typename S1, typename St>
  void dyson<G, S1, St>::find_mu(Sigma1& sigma1, Sigma_tau& sigma_tau_s) {
    double         mu = _mu;
    double         nel, nel1, nel2, nel_diff1, nel_diff2;
    double         mu1;
    double         mu2;
    double         delta = 0.5;
    utils::timing& t     = utils::timing::get_instance();
    t.start("Chemical potential search");

    std::cout << std::scientific << std::setprecision(15);
    mu = _mu;
    std::vector<std::complex<double>> eigenvalues_Sigma_p_F;
    selfenergy_eigenspectra(sigma1, sigma_tau_s, eigenvalues_Sigma_p_F);
    // Start search for the chemical potential
    nel = compute_number_of_electrons(mu, eigenvalues_Sigma_p_F);
    if (!utils::context.global_rank) std::cout << "nel:" << nel << " mu: " << mu << " target nel:" << _nel << std::endl;

    if (std::abs((nel - _nel) / double(_nel)) > _tol) {
      if (nel > _nel) {
        mu1      = mu - delta;
        double d = delta;
        do {
          nel1 = compute_number_of_electrons(mu1, eigenvalues_Sigma_p_F);
          if (!utils::context.global_rank) std::cout << "nel:" << nel1 << " mu: " << mu1 << std::endl;
          mu1 -= d;
        } while (nel1 > _nel);
        mu2       = mu;
        nel2      = nel;
        nel_diff1 = _nel - nel1;
        nel_diff2 = _nel - nel2;
      } else {
        mu2      = mu + delta;
        double d = delta;
        do {
          nel2 = compute_number_of_electrons(mu2, eigenvalues_Sigma_p_F);
          if (!utils::context.global_rank) std::cout << "nel:" << nel2 << " mu: " << mu2 << std::endl;
          mu2 += d;
        } while (nel2 < _nel);
        mu1       = mu;
        nel1      = nel;
        nel_diff1 = _nel - nel1;
        nel_diff2 = _nel - nel2;
      }
      while (std::abs((nel - _nel) / double(_nel)) > _tol && std::abs(mu2 - mu1) > 0.01 * _tol) {
        mu  = (mu1 + mu2) * 0.5;
        nel = compute_number_of_electrons(mu, eigenvalues_Sigma_p_F);
        if (nel > _nel) {
          mu2 = mu;
        } else {
          mu1 = mu;
        }
        if (!utils::context.global_rank) std::cout << "nel:" << nel << " mu: " << mu << std::endl;
      }
    }
    if (!utils::context.global_rank) std::cout << "Chemical potential has been found. mu = " << mu << std::endl;
    _nel_found = nel;
    _mu        = mu;
    t.end();
  }

  template <>
  void dyson<utils::shared_object<ztensor<5>>, ztensor<4>, utils::shared_object<ztensor<5>>>::compute_G(
      G& g_tau_s, Sigma1& sigma1, Sigma_tau& sigma_tau_s) const {
    // Get G(tau)
    auto&                       g_tau         = g_tau_s.object();
    auto&                       sigma_tau     = sigma_tau_s.object();
    MPI_Datatype                dt_matrix     = utils::create_matrix_datatype<std::complex<double>>(_nso * _nso);
    MPI_Op                      matrix_sum_op = utils::create_matrix_operation<std::complex<double>>();
    ztensor<3>                  G_t(_nts, _nso, _nso);
    ztensor<3>                  G_c(1, _nso, _nso);
    ztensor<3>                  Sigma_c(_ncheb, _nso, _nso);
    ztensor<3>                  G_w(_nw, _nso, _nso);
    ztensor<3>                  Sigma_w(_nw, _nso, _nso);
    ztensor<3>                  Sigma_k(_nts, _nso, _nso);
    Eigen::FullPivLU<MatrixXcd> lusolver(_nso, _nso);
    g_tau_s.fence();
    if (!utils::context.node_rank) g_tau.set_zero();
    double coeff_last  = 0.0;
    double coeff_first = 0.0;
    g_tau_s.fence();
    g_tau_s.fence();
    for (int isk = utils::context.global_rank; isk < _ns * _ink; isk += utils::context.global_size) {
      int is = isk / _ink;
      int ik = isk % _ink;
      Sigma_k.set_zero();
      // G_w.set_zero();
      for (int it = 0; it < _nts; ++it) matrix(Sigma_k(it)) = matrix(sigma_tau(it, is, ik));
      _ft.tau_to_omega(Sigma_k, Sigma_w, 1);
      for (int ic = 0; ic < _nw; ++ic) {
        std::complex<double> muomega = _ft.wsample_fermi()(ic) * 1.0i + _mu;
        matrix(G_w(ic)) = muomega * matrix(_S_k(is, ik)) - matrix(_H_k(is, ik)) - matrix(sigma1(is, ik)) - matrix(Sigma_w(ic));
        matrix(G_w(ic)) = lusolver.compute(matrix(G_w(ic))).inverse().eval();
      }

      // Transform back to tau
      _ft.omega_to_tau(G_w, G_t, 1);

      for (int it = 0; it < _nts; ++it) {
        matrix(g_tau(it, is, ik)) = matrix(G_t(it));
      }
      // Check Chebyshev
      _ft.tau_to_chebyshev_c(G_t, G_c, _ncheb - 1, 1);
      coeff_last = std::max(matrix(G_c(0)).cwiseAbs().maxCoeff(), coeff_last);
      _ft.tau_to_chebyshev_c(G_t, G_c, 0, 1);
      coeff_first = std::max(matrix(G_c(0)).cwiseAbs().maxCoeff(), coeff_first);
    }
    g_tau_s.fence();
    g_tau_s.fence();
    if (!utils::context.node_rank) {
      utils::allreduce(MPI_IN_PLACE, g_tau.data(), g_tau.size() / (_nso * _nso), dt_matrix, matrix_sum_op,
                       utils::context.internode_comm);
    }
    g_tau_s.fence();
    double leakage = coeff_last / coeff_first;
    if (!utils::context.global_rank) std::cout << "Leakage of Dyson G: " << leakage << std::endl;
    if (!utils::context.global_rank and leakage > 1e-8) std::cerr << "Warning: The leakage is larger than 1e-8"<<std::endl;
    MPI_Type_free(&dt_matrix);
    MPI_Op_free(&matrix_sum_op);
  }

  template <>
  void dyson<ztensor<5>, ztensor<4>, ztensor<5>>::compute_G(G& g_tau, Sigma1& sigma1, Sigma_tau& sigma_tau) const {
    // Get G(tau)
    MPI_Datatype                dt_matrix     = utils::create_matrix_datatype<std::complex<double>>(_nso * _nso);
    MPI_Op                      matrix_sum_op = utils::create_matrix_operation<std::complex<double>>();
    ztensor<3>                  G_t(_nts, _nso, _nso);
    ztensor<3>                  G_c(1, _nso, _nso);
    ztensor<3>                  Sigma_c(_ncheb, _nso, _nso);
    ztensor<3>                  G_w(_nw, _nso, _nso);
    ztensor<3>                  Sigma_w(_nw, _nso, _nso);
    ztensor<3>                  Sigma_k(_nts, _nso, _nso);
    Eigen::FullPivLU<MatrixXcd> lusolver(_nso, _nso);
    if (!utils::context.node_rank) g_tau.set_zero();
    double coeff_last  = 0.0;
    double coeff_first = 0.0;
    for (int isk = utils::context.global_rank; isk < _ns * _ink; isk += utils::context.global_size) {
      int is = isk / _ink;
      int ik = isk % _ink;
      Sigma_k.set_zero();
      // G_w.set_zero();
      for (int it = 0; it < _nts; ++it) matrix(Sigma_k(it)) = matrix(sigma_tau(it, is, ik));
      _ft.tau_to_omega(Sigma_k, Sigma_w, 1);
      for (int ic = 0; ic < _nw; ++ic) {
        std::complex<double> muomega = _ft.wsample_fermi()(ic) * 1.0i + _mu;
        matrix(G_w(ic)) = muomega * matrix(_S_k(is, ik)) - matrix(_H_k(is, ik)) - matrix(sigma1(is, ik)) - matrix(Sigma_w(ic));
        matrix(G_w(ic)) = lusolver.compute(matrix(G_w(ic))).inverse().eval();
        // G_w(ic).matrix() = G_w(ic).matrix().inverse();
      }

      // Transform back to tau
      _ft.omega_to_tau(G_w, G_t, 1);

      // Eigen::ComplexEigenSolver<MatrixXcd> solver(_nso);
      for (int it = 0; it < _nts; ++it) {
        matrix(g_tau(it, is, ik)) = matrix(G_t(it));
      }
      // Check Chebyshev
      _ft.tau_to_chebyshev_c(G_t, G_c, _ncheb - 1, 1);
      coeff_last = std::max(matrix(G_c(0)).cwiseAbs().maxCoeff(), coeff_last);
      _ft.tau_to_chebyshev_c(G_t, G_c, 0, 1);
      coeff_first = std::max(matrix(G_c(0)).cwiseAbs().maxCoeff(), coeff_first);
    }
    if (!utils::context.node_rank) {
      utils::allreduce(MPI_IN_PLACE, g_tau.data(), g_tau.size() / (_nso * _nso), dt_matrix, matrix_sum_op,
                       utils::context.internode_comm);
    }
    double leakage = coeff_last / coeff_first;
    if (!utils::context.global_rank) std::cout << "Leakage of Dyson G: " << leakage << std::endl;
    if (!utils::context.global_rank and leakage > 1e-8) std::cerr << "Warning: The leakage is larger than 1e-8"<<std::endl;
    MPI_Type_free(&dt_matrix);
    MPI_Op_free(&matrix_sum_op);
  }

  template <>
  double dyson<utils::shared_object<ztensor<5>>, ztensor<4>, utils::shared_object<ztensor<5>>>::diff(G& g, Sigma1& sigma1,
                                                                                                     Sigma_tau& sigma_tau) {
    compute_G(g, sigma1, sigma_tau);
    auto [e1, e2, e3] = compute_energy(g.object(), sigma1, sigma_tau.object(), _H_k, _ft, _bz_utils, _nao != _nso);
    double diff       = std::abs(_E_1b - e1) + std::abs(_E_hf - e2) + std::abs(_E_corr - e3);
    _E_1b             = e1;
    _E_hf             = e2;
    _E_corr           = e3;
    return diff;
  }

  template <>
  double dyson<ztensor<5>, ztensor<4>, ztensor<5>>::diff(G& g, Sigma1& sigma1, Sigma_tau& sigma_tau) {
    compute_G(g, sigma1, sigma_tau);
    auto [e1, e2, e3] = compute_energy(g, sigma1, sigma_tau, _H_k, _ft, _bz_utils, _nao != _nso);
    double diff       = std::abs(_E_1b - e1) + std::abs(_E_hf - e2) + std::abs(_E_corr - e3);
    _E_1b             = e1;
    _E_hf             = e2;
    _E_corr           = e3;
    return diff;
  }

  template <typename G, typename S1, typename St>
  void dyson<G, S1, St>::solve(G& g, Sigma1& sigma1, Sigma_tau& sigma_tau) {
    if (_const_density) {
      find_mu(sigma1, sigma_tau);
    }
    compute_G(g, sigma1, sigma_tau);
  }

  template <typename G, typename S1, typename St>
  void dyson<G, S1, St>::dump_iteration(size_t iter, const std::string& result_file) {
    if (!utils::context.global_rank) {
      h5pp::archive ar(result_file, "a");
      ar["iter" + std::to_string(iter) + "/G_tau/mesh"] << _ft.sd().repn_fermi().tsample();
      ar["iter" + std::to_string(iter) + "/Selfenergy/mesh"] << _ft.sd().repn_fermi().tsample();
      ar["iter" + std::to_string(iter) + "/Energy_1b"] << _E_1b;
      ar["iter" + std::to_string(iter) + "/Energy_HF"] << _E_hf + _E_nuc;
      ar["iter" + std::to_string(iter) + "/Energy_2b"] << _E_corr;
      ar.close();
      std::cout << "One-body Energy: " << _E_1b << std::endl;
      std::cout << "HF Energy: " << _E_hf + _E_nuc << std::endl;
      std::cout << "Correlation Energy: " << _E_corr << std::endl;
    }
  }

  template class dyson<utils::shared_object<ztensor<5>>, ztensor<4>, utils::shared_object<ztensor<5>>>;
  template class dyson<ztensor<5>, ztensor<4>, ztensor<5>>;
}  // namespace green::mbpt
