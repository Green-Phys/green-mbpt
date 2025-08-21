/*
 * Copyright (c) 2023 University of Michigan
 *
 */

#include <green/grids.h>
#include <green/mbpt/dyson.h>
#include <green/mbpt/gf2_solver.h>
#include <green/mbpt/gw_solver.h>
#include <green/mbpt/hf_solver.h>
#include <green/sc/sc_loop.h>
#include <green/mbpt/mbpt_run.h>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include "catch2/matchers/catch_matchers.hpp"
#include "tensor_test.h"

void solve_hf(const std::string& input, const std::string& int_hf, const std::string& data) {
  auto        p           = green::params::params("DESCR");
  std::string input_file  = TEST_PATH + input;
  std::string df_int_path = TEST_PATH + int_hf;
  std::string test_file   = TEST_PATH + data;
  std::string grid_file   = GRID_PATH + "/ir/1e4.h5"s;
  std::string args =
      "test --restart 0 --itermax 1 --E_thr 1e-13 --mixing_type SIGMA_MIXING --mixing_weight 0.8 --input_file=" + input_file +
      " --BETA 100 --grid_file=" + grid_file + " --dfintegral_hf_file=" + df_int_path;
  green::grids::define_parameters(p);
  green::mbpt::define_parameters(p);
  green::symmetry::define_parameters(p);
  p.parse(args);
  green::symmetry::brillouin_zone_utils bz(p);
  size_t                                nso, ns, nk, ink, nts;
  green::sc::ztensor<4>                 tmp;
  {
    green::h5pp::archive ar(input_file);
    ar["params/nso"] >> nso;
    ar["params/ns"] >> ns;
    ar["params/nk"] >> nk;
    ar["grid/ink"] >> ink;
    green::sc::dtensor<5> S_k;
    ar["HF/S-k"] >> S_k;
    ar.close();
    tmp.resize(ns, nk, nso, nso);
    tmp << S_k.view<std::complex<double>>().reshape(ns, nk, nso, nso);
  }
  {
    green::h5pp::archive ar(grid_file);
    ar["fermi/metadata/ncoeff"] >> nts;
    ar.close();
    nts += 2;
  }
  auto G_shared    = green::utils::shared_object(green::sc::ztensor<5>(nullptr, nts, ns, ink, nso, nso));
  auto S_shared    = green::utils::shared_object(green::sc::ztensor<5>(nullptr, nts, ns, ink, nso, nso));
  auto Sigma1      = green::sc::ztensor<4>(ns, ink, nso, nso);
  auto Sigma1_test = green::sc::ztensor<4>(ns, ink, nso, nso);
  auto Sk          = green::sc::ztensor<4>(ns, ink, nso, nso);
  for (int is = 0; is < ns; ++is) Sk(is) << bz.full_to_ibz(tmp(is));
  {
    green::h5pp::archive ar(test_file, "r");
    G_shared.fence();
    if (!green::utils::context.node_rank) ar["G_tau"] >> G_shared.object();
    G_shared.fence();
    ar["result/Sigma1"] >> Sigma1_test;
    ar.close();
  }
  green::mbpt::hf_solver solver(p, bz, Sk);
  solver.solve(G_shared, Sigma1, S_shared);
  REQUIRE_THAT(Sigma1, IsCloseTo(Sigma1_test));
}

void solve_gw(const std::string& input, const std::string& int_f, const std::string& data, const std::string& q0 = "IGNORE_G0") {
  auto        p           = green::params::params("DESCR");
  std::string input_file  = TEST_PATH + input;
  std::string df_int_path = TEST_PATH + int_f;
  std::string test_file   = TEST_PATH + data;
  std::string grid_file   = GRID_PATH + "/ir/1e4.h5"s;
  std::string args =
      "test --restart 0 --itermax 1 --E_thr 1e-13 --mixing_type SIGMA_MIXING --mixing_weight 0.8 --input_file=" + input_file +
      " --BETA 100 --grid_file=" + grid_file + " --dfintegral_file=" + df_int_path + " --q0_treatment " + q0;
  green::grids::define_parameters(p);
  green::mbpt::define_parameters(p);
  green::symmetry::define_parameters(p);
  p.define<double>("BETA", "Inverse temperature", 10.0);
  p.parse(args);
  green::symmetry::brillouin_zone_utils bz(p);
  green::grids::transformer_t           ft(p);
  size_t                                nso, ns, nk, ink, nts;
  green::sc::ztensor<4>                 tmp;
  {
    green::h5pp::archive ar(input_file);
    ar["params/nso"] >> nso;
    ar["params/ns"] >> ns;
    ar["params/nk"] >> nk;
    ar["grid/ink"] >> ink;
    green::sc::dtensor<5> S_k;
    ar["HF/S-k"] >> S_k;
    ar.close();
    tmp.resize(ns, nk, nso, nso);
    tmp << S_k.view<std::complex<double>>().reshape(ns, nk, nso, nso);
  }
  {
    green::h5pp::archive ar(grid_file);
    ar["fermi/metadata/ncoeff"] >> nts;
    ar.close();
    nts += 2;
  }
  auto G_shared     = green::utils::shared_object(green::sc::ztensor<5>(nullptr, nts, ns, ink, nso, nso));
  auto S_shared     = green::utils::shared_object(green::sc::ztensor<5>(nullptr, nts, ns, ink, nso, nso));
  auto S_shared_tst = green::utils::shared_object(green::sc::ztensor<5>(nullptr, nts, ns, ink, nso, nso));
  auto Sigma1       = green::sc::ztensor<4>(ns, ink, nso, nso);
  auto Sk           = green::sc::ztensor<4>(ns, ink, nso, nso);
  for (int is = 0; is < ns; ++is) Sk(is) << bz.full_to_ibz(tmp(is));
  {
    green::h5pp::archive ar(test_file, "r");
    G_shared.fence();
    if (!green::utils::context.node_rank) ar["G_tau"] >> G_shared.object();
    G_shared.fence();
    S_shared_tst.fence();
    if (!green::utils::context.node_rank) ar["result/Sigma_tau"] >> S_shared_tst.object();
    S_shared_tst.fence();
    ar.close();
  }
  green::mbpt::gw_solver solver(p, ft, bz, Sk);
  solver.solve(G_shared, Sigma1, S_shared);
  REQUIRE_THAT(S_shared.object(), IsCloseTo(S_shared_tst.object(), 1e-6));
}

void solve_gf2(const std::string& df_int_path, const std::string& test_file, const std::string& input_file) {
  auto        p         = green::params::params("DESCR");

  std::string grid_file = GRID_PATH + "/ir/1e4.h5"s;
  std::string args =
      "test --restart 0 --itermax 1 --E_thr 1e-13 --mixing_type SIGMA_MIXING --mixing_weight 0.8 --input_file=" + input_file +
      " --BETA 100 --grid_file=" + grid_file + " --dfintegral_file=" + df_int_path + " --dfintegral_hf_file=" + df_int_path;
  green::grids::define_parameters(p);
  green::mbpt::define_parameters(p);
  green::symmetry::define_parameters(p);
  p.parse(args);
  green::symmetry::brillouin_zone_utils bz(p);
  green::grids::transformer_t           ft(p);
  size_t                                nso, ns, nk, ink, nts;
  green::sc::ztensor<4>                 tmp;
  {
    green::h5pp::archive ar(input_file);
    ar["params/nso"] >> nso;
    ar["params/ns"] >> ns;
    ar["params/nk"] >> nk;
    ar["grid/ink"] >> ink;
    green::sc::dtensor<5> S_k;
    ar["HF/S-k"] >> S_k;
    ar.close();
    tmp.resize(ns, nk, nso, nso);
    tmp << S_k.view<std::complex<double>>().reshape(ns, nk, nso, nso);
  }
  {
    green::h5pp::archive ar(grid_file);
    ar["fermi/metadata/ncoeff"] >> nts;
    ar.close();
    nts += 2;
  }
  auto G_shared     = green::utils::shared_object(green::sc::ztensor<5>(nullptr, nts, ns, ink, nso, nso));
  auto S_shared     = green::utils::shared_object(green::sc::ztensor<5>(nullptr, nts, ns, ink, nso, nso));
  auto S_shared_tst = green::utils::shared_object(green::sc::ztensor<5>(nullptr, nts, ns, ink, nso, nso));
  auto Sigma1       = green::sc::ztensor<4>(ns, ink, nso, nso);
  auto Sk           = green::sc::ztensor<4>(ns, ink, nso, nso);
  for (int is = 0; is < ns; ++is) Sk(is) << bz.full_to_ibz(tmp(is));
  {
    green::h5pp::archive ar(test_file, "r");
    G_shared.fence();
    if (!green::utils::context.node_rank) ar["G_tau"] >> G_shared.object();
    G_shared.fence();
    S_shared_tst.fence();
    if (!green::utils::context.node_rank) ar["result/Sigma_tau"] >> S_shared_tst.object();
    S_shared_tst.fence();
    ar.close();
  }
  green::mbpt::gf2_solver solver(p, ft, bz);
  solver.solve(G_shared, Sigma1, S_shared);
  REQUIRE_THAT(S_shared.object(), IsCloseTo(S_shared_tst.object(), 1e-6));
}

TEST_CASE("MBPT Solver") {
  SECTION("HF") { solve_hf("/HF/input.h5", "/HF/df_hf_int", "/HF/data.h5"); }
  SECTION("HF_X2C") { solve_hf("/HF_X2C/input.h5", "/HF_X2C/df_hf_int", "/HF_X2C/data.h5"); }
  SECTION("GW") { solve_gw("/GW/input.h5", "/GW/df_int", "/GW/data.h5"); }
  SECTION("GW_C") { solve_gw("/GW/input.h5", "/GW/df_hf_int", "/GW/data_c.h5", "EXTRAPOLATE"); }
  SECTION("GW_X2C") { solve_gw("/GW_X2C/input.h5", "/GW_X2C/df_hf_int", "/GW_X2C/data.h5"); }

  SECTION("GF2") { solve_gf2(TEST_PATH + "/GF2/df_hf_int"s, TEST_PATH + "/GF2/data.h5"s, TEST_PATH + "/GF2/input.h5"s); }

  SECTION("GF2_Empty_Ewald") {
    solve_gf2(TEST_PATH + "/GF2/df_hf_int_z"s, TEST_PATH + "/GF2/data.h5"s, TEST_PATH + "/GF2/input.h5"s);
  }

  SECTION("GF2_Ewald") {
    solve_gf2(TEST_PATH + "/GF2/df_hf_int_e"s, TEST_PATH + "/GF2/data_e.h5"s, TEST_PATH + "/GF2/input_e.h5"s);
  }

  SECTION("Input Data Version") {
    auto        p             = green::params::params("DESCR");
    std::string input_file    = TEST_PATH + "/Input/input.h5"s;
    std::string df_int_path_1 = TEST_PATH + "/Input/df_int"s;
    std::string df_int_path_2 = TEST_PATH + "/Input/df_int_x"s;
    std::string df_int_path_3 = TEST_PATH + "/Input/df_int_y"s;
    std::string df_int_path_4 = TEST_PATH + "/Input/df_int_0.3.0"s;
    std::string grid_file     = GRID_PATH + "/ir/1e4.h5"s;
    std::string args =
        "test --restart 0 --itermax 2 --E_thr 1e-13 --mixing_type G_MIXING --mixing_weight 0.8 --input_file=" + input_file +
        " --BETA 100 --verbose=1 --grid_file=" + grid_file + " --dfintegral_file=" + df_int_path_1 +
        " --dfintegral_hf_file=" + df_int_path_2;
    green::sc::define_parameters(p);
    green::symmetry::define_parameters(p);
    green::grids::define_parameters(p);
    green::mbpt::define_parameters(p);
    p.parse(args);
    green::symmetry::brillouin_zone_utils<green::symmetry::inv_symm_op> bz_utils(p);
    REQUIRE_THROWS_AS(green::mbpt::df_integral_t(df_int_path_1, 2, 36, bz_utils), green::mbpt::mbpt_outdated_input);
    REQUIRE_THROWS_AS(green::mbpt::df_integral_t(df_int_path_2, 2, 36, bz_utils), green::mbpt::mbpt_outdated_input);
    REQUIRE_NOTHROW(green::mbpt::df_integral_t(df_int_path_3, 2, 36, bz_utils));
    REQUIRE_NOTHROW(green::mbpt::df_integral_t(df_int_path_4, 2, 36, bz_utils));
    REQUIRE_THROWS_AS(green::mbpt::check_input(p), green::mbpt::mbpt_outdated_input);
  }

  SECTION("Test Version Strings") {
    std::vector<std::string> fail_versions = {"0.2.0", "0.2.3"};
    std::vector<std::string> pass_versions = {
      "0.2.4", "0.2.4b10", "0.3.0", "0.3.0b8", "0.3.1", "0.3.1b10"
    };
    for (int i=0; i < fail_versions.size() - 1; i++) {
      REQUIRE_FALSE(green::mbpt::CheckVersion(fail_versions[i]));
    }
    for (int i=0; i < pass_versions.size() - 1; i++) {
      REQUIRE(green::mbpt::CheckVersion(pass_versions[i]));
    }
  }

  SECTION("Init real Dyson") {
    auto        p          = green::params::params("DESCR");
    std::string input_file = TEST_PATH + "/Dyson/input.h5"s;
    std::string grid_file  = GRID_PATH + "/ir/1e4.h5"s;
    std::string args =
        "test --restart 0 --itermax 2 --E_thr 1e-13 --mixing_type G_MIXING --mixing_weight 0.8 --input_file=" + input_file +
        " --BETA 100 --verbose=1 --grid_file=" + grid_file;
    green::sc::define_parameters(p);
    green::symmetry::define_parameters(p);
    green::grids::define_parameters(p);
    green::mbpt::define_parameters(p);
    p.parse(args);
    green::sc::noop_solver                            noop;
    green::sc::sc_loop<green::mbpt::shared_mem_dyson> sc(MPI_COMM_WORLD, p);
    size_t                                            nts;
    size_t                                            ns  = sc.dyson_solver().ns();
    size_t                                            nk  = sc.dyson_solver().bz_utils().nk();
    size_t                                            ink = sc.dyson_solver().bz_utils().ink();
    size_t                                            nao = sc.dyson_solver().nao();
    size_t                                            nso = sc.dyson_solver().nso();
    green::sc::ztensor<4>                             tmp;
    {
      green::h5pp::archive  ar(input_file);
      green::sc::dtensor<5> H_k;
      green::sc::dtensor<5> F_k;
      ar["HF/H-k"] >> H_k;
      ar["HF/Fock-k"] >> F_k;
      ar.close();
      F_k -= H_k;
      tmp.resize(ns, nk, nao, nao);
      tmp << F_k.view<std::complex<double>>().reshape(ns, nk, nao, nao);
    }
    nts = sc.dyson_solver().ft().sd().repn_fermi().nts();
    green::symmetry::brillouin_zone_utils bz(p);
    auto G_shared = green::utils::shared_object(green::sc::ztensor<5>(nullptr, nts, ns, ink, nao, nao));
    auto S_shared = green::utils::shared_object(green::sc::ztensor<5>(nullptr, nts, ns, ink, nao, nao));
    auto Sigma1   = green::sc::ztensor<4>(ns, ink, nao, nao);
    S_shared.fence();
    if (!green::utils::context.node_rank) S_shared.object().set_zero();
    S_shared.fence();
    Sigma1(0) << sc.dyson_solver().bz_utils().full_to_ibz(tmp(0));
    Sigma1(1) << sc.dyson_solver().bz_utils().full_to_ibz(tmp(1));
    sc.dyson_solver().mu() = -1.5;
    sc.solve(noop, sc.dyson_solver().H_k(), sc.dyson_solver().S_k(), G_shared, Sigma1, S_shared);
  }
  SECTION("Init real Dyson. non shared") {
    auto        p          = green::params::params("DESCR");
    std::string input_file = TEST_PATH + "/Dyson/input.h5"s;
    std::string grid_file  = GRID_PATH + "/ir/1e4.h5"s;
    std::string args =
        "test --restart 0 --itermax 2 --E_thr 1e-13 --mixing_type G_MIXING --mixing_weight 0.8 --input_file=" + input_file +
        " --BETA 100 --grid_file=" + grid_file;
    green::sc::define_parameters(p);
    green::symmetry::define_parameters(p);
    green::grids::define_parameters(p);
    green::mbpt::define_parameters(p);
    p.parse(args);
    green::sc::noop_solver                           noop;
    green::sc::sc_loop<green::mbpt::local_mem_dyson> sc(MPI_COMM_WORLD, p);
    size_t                                           nts, ns, nk, ink, nao;
    green::sc::ztensor<4>                            tmp;
    {
      green::h5pp::archive ar(input_file);
      ar["params/nao"] >> nao;
      ar["params/ns"] >> ns;
      ar["params/nk"] >> nk;
      ar["grid/ink"] >> ink;
      green::sc::dtensor<5> H_k;
      green::sc::dtensor<5> F_k;
      ar["HF/H-k"] >> H_k;
      ar["HF/Fock-k"] >> F_k;
      ar.close();
      F_k -= H_k;
      tmp.resize(ns, nk, nao, nao);
      tmp << F_k.view<std::complex<double>>().reshape(ns, nk, nao, nao);
    }
    {
      green::h5pp::archive ar(grid_file);
      ar["fermi/metadata/ncoeff"] >> nts;
      ar.close();
      nts += 2;
    }
    green::symmetry::brillouin_zone_utils bz(p);
    auto                                  G      = green::sc::ztensor<5>(nts, ns, ink, nao, nao);
    auto                                  S      = green::sc::ztensor<5>(nts, ns, ink, nao, nao);
    auto                                  Sigma1 = green::sc::ztensor<4>(ns, ink, nao, nao);
    Sigma1(0) << bz.full_to_ibz(tmp(0));
    Sigma1(1) << bz.full_to_ibz(tmp(1));
    sc.solve(noop, sc.dyson_solver().H_k(), sc.dyson_solver().S_k(), G, Sigma1, S);
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
