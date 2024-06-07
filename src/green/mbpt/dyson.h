/*
 * Copyright (c) 2023 University of Michigan
 *
 */
#ifndef SC_DYSON_H
#define SC_DYSON_H

#include <green/grids.h>
#include <green/symmetry/symmetry.h>
#include <green/utils/mpi_shared.h>
#include <green/utils/mpi_utils.h>

#include "common_defs.h"

namespace green::mbpt {

  template <typename T>
  auto& extract_local(const T& t) {
    if constexpr (std::is_same_v<T, utils::shared_object<ztensor<5>>>) {
      return t.object();
    } else {
      return t;
    }
  }

  template <typename G_type, typename Sigma1_type, typename Sigma_tau_type>
  class dyson {
    using brillouin_zone_utils = symmetry::brillouin_zone_utils<symmetry::inv_symm_op>;

  public:
    using G         = G_type;
    using Sigma1    = Sigma1_type;
    using Sigma_tau = Sigma_tau_type;

    dyson(const params::params& p);

    virtual ~dyson(){};

    /**
     * Using Dyson equation compute new g_tau for given static (sigma1) and dynamic (sigma_tau) parts of the self-energy
     *
     * @param g_tau [OUT] new
     * @param sigma1
     * @param sigma_tau
     */
    void   compute_G(G& g_tau, Sigma1& sigma1, Sigma_tau& sigma_tau) const;

    /**
     * For each matsubara frequency compute eigenvalues of (H + sigma1 + sigma_w) to speedup chemical potetial search
     *
     * @param sigma1 - static part of the self-energy
     * @param sigma_tau - dynamic part of the self-energy
     * @param eigenvalues_Sigma_p_F - eigenvalues of (H + sigma1 + sigma_w)
     */
    void   selfenergy_eigenspectra(const Sigma1& sigma1, const Sigma_tau& sigma_tau,
                                   std::vector<std::complex<double>>& eigenvalues_Sigma_p_F) const;

    /**
     * For a given chemical potetial and eigenspectra of (H + sigma1 + sigma_w) find number of electrons
     *
     * @param mu - chemical potetial
     * @param eigenvalues_Sigma_p_F - eigenspectra of (H + sigma1 + sigma_w)
     * @return number of electrons for given parameters
     */
    double compute_number_of_electrons(double mu, const std::vector<std::complex<double>>& eigenvalues_Sigma_p_F) const;

    /**
     * From diagonalized Dyson equation find new chemical potential
     * @param sigma1 - static part of the self-energy
     * @param sigma_tau_s - dynamic part of the self-energy
     */
    std::pair<double, double>   find_mu(const Sigma1& sigma1, const Sigma_tau& sigma_tau_s) const;

    /**
     * Store additional information for current iteration
     *
     * @param iter - number of the current iteration
     * @param gtau - Green's function to be dumped
     * @param s1 - Static part of the self-energy to be dumped
     * @param st - Dynamic part of the self-energy to be dumped
     * @param result_file - name of the file with results
     */
    void                        dump_iteration(size_t iter, const G_type& gtau, const Sigma1_type&s1, const Sigma_tau_type&st,
                                               const std::string& result_file);

    /**
     * For a given static and dynamic parts of a self-energy solve the dyson equation and obtain new
     * Green's function. Number of electrons in the local unit-cell will be fixed
     * if the `const_density` parameter has been set to true.
     *
     * @param g - Green's function to be evaluated
     * @param sigma1 - static part of a self-energy
     * @param sigma_tau - dynamic part of a self-energy
     */
    void                        solve(G& g, Sigma1& sigma1, Sigma_tau& sigma_tau);

    /**
     * Compute difference for a target parameter for the current iteration
     *
     * @param g - Green's function at the current iteration
     * @param sigma1 - static part of a self-energy at the current iteration
     * @param sigma_tau - dynamic part of a self-energy at the current iteration
     */
    double                      diff(G& g, Sigma1& sigma1, Sigma_tau& sigma_tau);

    const grids::transformer_t& ft() const { return _ft; }
    //
    const brillouin_zone_utils& bz_utils() const { return _bz_utils; }
    const ztensor<4>&           S_k() const { return _S_k; }
    const ztensor<4>&           H_k() const { return _H_k; }

    [[nodiscard]] size_t        nao() const { return _nao; }
    [[nodiscard]] size_t        nso() const { return _nso; }
    [[nodiscard]] size_t        ns() const { return _ns; }
    [[nodiscard]] double        mu() const { return _mu; }
    double&                     mu() { return _mu; }

  protected:
    // Imaginary time/frequency transform class
    grids::transformer_t _ft;
    //
    brillouin_zone_utils _bz_utils;
    // number of Chebyshev polynomials
    int                  _ncheb;
    // number of tau points
    int                  _nts;
    // number of frequency points
    int                  _nw;
    // number of k-points
    int                  _nk;
    // number of k-points in the reduced Brillouin zone
    int                  _ink;
    // number of spin-orbitals
    int                  _nso;
    // number of orbitals
    int                  _nao;
    // number of spins
    int                  _ns;
    //
    bool                 _X2C;
    // chemical potential data
    double               _mu;
    // Number of electrons to be preserved during dyson equation solution
    double               _nel;
    // Current number of electrons
    double               _nel_found;
    // do we need to preserve number of electrons
    bool                 _const_density;
    //
    double               _tol;
    //
    ztensor<4>           _H_k;
    ztensor<4>           _S_k;
    //
    double               _E_corr{0};
    double               _E_hf{0};
    double               _E_1b{0};
    double               _E_nuc{0};

    int                  _verbose;

  private:
    void print_convergence(size_t iter, const G&gtau, const std::string& result_file);
  };

  using shared_mem_dyson = dyson<utils::shared_object<ztensor<5>>, ztensor<4>, utils::shared_object<ztensor<5>>>;
  using local_mem_dyson  = dyson<ztensor<5>, ztensor<4>, ztensor<5>>;
}  // namespace green::mbpt
#endif  // SC_DYSON_H
