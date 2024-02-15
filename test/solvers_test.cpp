/*
 * Copyright (c) 2023 University of Michigan
 *
 */

#include <green/grids.h>
#include <green/mbpt/dyson.h>
#include <green/mbpt/hf_solver.h>
#include <green/sc/sc_loop.h>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("MBPT Solver") {
  SECTION("HF") {
    auto        p           = green::params::params("DESCR");
    std::string input_file  = TEST_PATH + "/HF/input.h5"s;
    std::string df_int_path = TEST_PATH + "/HF/df_int"s;
    std::string grid_file   = GRID_PATH + "/ir/1e4.h5"s;
    std::string args =
        "test --restart 0 --itermax 1 --E_thr 1e-13 --mixing_type G_DAMPING --damping 0.8 --input_file=" + input_file +
        " --BETA 100 --grid_file=" + grid_file + " --dfintegral_hf_file=" + df_int_path;
    green::grids::define_parameters(p);
    green::mbpt::define_parameters(p);
    green::symmetry::define_parameters(p);
    p.define<double>("BETA", "Inverse temperature", 10.0);
    p.parse(args);
    green::symmetry::brillouin_zone_utils bz(p);
    size_t                                nao, ns, nk, ink, nts;
    green::sc::ztensor<4>                 tmp;
    green::sc::ztensor<4>                 tmp2;
    {
      green::h5pp::archive ar(input_file);
      ar["params/nao"] >> nao;
      ar["params/ns"] >> ns;
      ar["params/nk"] >> nk;
      ar["grid/ink"] >> ink;
      green::sc::dtensor<5> S_k;
      green::sc::dtensor<5> H_k;
      green::sc::dtensor<5> F_k;
      ar["HF/S-k"] >> S_k;
      ar["HF/H-k"] >> H_k;
      ar["HF/Fock-k"] >> F_k;
      ar.close();
      F_k -= H_k;
      tmp.resize(ns, nk, nao, nao);
      tmp2.resize(ns, nk, nao, nao);
      tmp << F_k.view<std::complex<double>>().reshape(ns, nk, nao, nao);
      tmp2 << S_k.view<std::complex<double>>().reshape(ns, nk, nao, nao);
    }
    {
      green::h5pp::archive ar(grid_file);
      ar["fermi/metadata/ncoeff"] >> nts;
      ar.close();
      nts += 2;
    }
    auto G_shared = green::utils::shared_object(green::sc::ztensor<5>(nullptr, nts, ns, ink, nao, nao));
    auto S_shared = green::utils::shared_object(green::sc::ztensor<5>(nullptr, nts, ns, ink, nao, nao));
    auto Sigma1   = green::sc::ztensor<4>(ns, ink, nao, nao);
    auto Sk       = green::sc::ztensor<4>(ns, ink, nao, nao);
    Sigma1(0) << bz.full_to_ibz(tmp2(0));
    Sigma1(1) << bz.full_to_ibz(tmp2(1));
    Sk(0) << bz.full_to_ibz(tmp(0));
    Sk(1) << bz.full_to_ibz(tmp(1));
    green::mbpt::hf_solver solver(p, bz, Sk);
  }

  SECTION("GW") {}

  SECTION("GF2") {}

  SECTION("Init real Dyson") {
    auto        p          = green::params::params("DESCR");
    std::string input_file = TEST_PATH + "/Dyson/input.h5"s;
    std::string grid_file  = GRID_PATH + "/ir/1e4.h5"s;
    std::string args =
        "test --restart 0 --itermax 1 --E_thr 1e-13 --mixing_type G_DAMPING --damping 0.8 --input_file=" + input_file +
        " --BETA 100 --grid_file=" + grid_file;
    green::sc::define_parameters(p);
    green::symmetry::define_parameters(p);
    green::grids::define_parameters(p);
    green::mbpt::define_parameters(p);
    p.parse(args);
    green::sc::noop_solver                            noop;
    green::sc::sc_loop<green::mbpt::shared_mem_dyson> sc(MPI_COMM_WORLD, p);
    size_t                                            nts, ns, nk, ink, nao;
    green::sc::ztensor<4>                             tmp;
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
    auto G_shared = green::utils::shared_object(green::sc::ztensor<5>(nullptr, nts, ns, ink, nao, nao));
    auto S_shared = green::utils::shared_object(green::sc::ztensor<5>(nullptr, nts, ns, ink, nao, nao));
    auto Sigma1   = green::sc::ztensor<4>(ns, ink, nao, nao);
    S_shared.fence();
    if(!green::utils::context.node_rank) S_shared.object().set_zero();
    S_shared.fence();
    Sigma1(0) << bz.full_to_ibz(tmp(0));
    Sigma1(1) << bz.full_to_ibz(tmp(1));
    sc.solve(noop, G_shared, Sigma1, S_shared);
  }
  SECTION("Init real Dyson. non shared") {
    auto        p          = green::params::params("DESCR");
    std::string input_file = TEST_PATH + "/Dyson/input.h5"s;
    std::string grid_file  = GRID_PATH + "/ir/1e4.h5"s;
    std::string args =
        "test --restart 0 --itermax 1 --E_thr 1e-13 --mixing_type G_DAMPING --damping 0.8 --input_file=" + input_file +
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
    sc.solve(noop, G, Sigma1, S);
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
