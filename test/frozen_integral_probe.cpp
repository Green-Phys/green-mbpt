// Invoke production kernels on one supplied, unreduced frozen Green's function.
// This executable is a validation tool; it never runs a self-consistency loop.
#include <green/h5pp/archive.h>
#include <green/mbpt/kernel_factory.h>
#include <green/mbpt/mbpt_run.h>
#include <mpi.h>

#include <filesystem>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  auto& context = green::utils::context();
  int   status  = 0;
  try {
    using namespace green;
    auto p = params::params("Frozen integral kernel validation");
    sc::define_parameters(p);
    symmetry::define_parameters(p);
    grids::define_parameters(p);
    mbpt::define_parameters(p);
    p.define<std::string>("frozen_G_file", "Frozen G in interleaved (tau,spin,k,i,j,2) layout");
    p.define<std::string>("probe_output", "Fresh validation result file");
    if (!p.parse(argc, argv)) {
      if (!context.global_rank) p.help_or_version();
      MPI_Finalize();
      return 0;
    }
    mbpt::check_input(p);
    symmetry::brillouin_zone_utils bz(p);
    grids::transformer_t           ft(p);
    size_t                         nao, nso, ns, naux;
    double                         madelung;
    mbpt::ztensor<4>               overlap;
    {
      h5pp::archive input(p["input_file"]);
      input["params/nao"] >> nao;
      input["params/nso"] >> nso;
      input["params/ns"] >> ns;
      input["params/NQ"] >> naux;
      input["HF/madelung"] >> madelung;
      mbpt::ztensor<4> full;
      input["HF/S-k"] >> full;
      if (full.shape() != std::array<size_t, 4>{ns, bz.nk(), nso, nso})
        throw std::runtime_error("Overlap dimensions differ from input");
      overlap.resize(ns, bz.ink(), nso, nso);
      for (size_t s = 0; s < ns; ++s) overlap(s) << bz.full_to_ibz(full(s));
    }
    if (bz.ink() != bz.nk()) throw std::runtime_error("Frozen validation requires unreduced one-body k points");
    const size_t                           nts = ft.sd().repn_fermi().nts();
    utils::shared_object<mbpt::ztensor<5>> g(nts, ns, bz.ink(), nso, nso);
    utils::shared_object<mbpt::ztensor<5>> sigma(nts, ns, bz.ink(), nso, nso);
    g.fence();
    if (!context.node_rank) {
      h5pp::archive    file(p["frozen_G_file"]);
      mbpt::dtensor<6> raw;
      file["G_tau"] >> raw;
      if (raw.shape() != std::array<size_t, 6>{nts, ns, bz.nk(), nso, nso, 2})
        throw std::runtime_error("Frozen G dimensions differ from grid/input");
      g.object() << raw.view<std::complex<double>>().reshape(nts, ns, bz.nk(), nso, nso);
      sigma.object().set_zero();
    }
    g.fence();
    sigma.fence();
    const std::vector<std::complex<double>> before(g.object().data(), g.object().data() + g.object().size());
    mbpt::ztensor<4>                        sigma1(ns, bz.ink(), nso, nso);
    sigma1.set_zero();
    mbpt::hf_solver hf(p, bz, overlap);
    hf.solve(g, sigma1, sigma);
    const auto theory = p["scf_type"].as<mbpt::scf_type>();
    if (theory == mbpt::GW) {
      mbpt::gw_solver gw(p, ft, bz, overlap);
      gw.solve(g, sigma1, sigma);
    } else if (theory == mbpt::GF2) {
      // GF2 correlation remains on the CPU, independently of the HF kernel.
      mbpt::gf2_solver gf2(p, ft, bz);
      gf2.solve(g, sigma1, sigma);
    }
    for (size_t i = 0; i < before.size(); ++i)
      if (before[i] != g.object().data()[i]) throw std::runtime_error("Kernel mutated frozen G");
    if (!context.global_rank) {
      const std::string output = p["probe_output"];
      if (std::filesystem::exists(output)) throw std::runtime_error("Probe output must be fresh");
      h5pp::archive file(output, "w");
      file["Sigma1"] << sigma1;
      file["Selfenergy"] << sigma.object();
      std::cout << "Frozen integral kernel validation completed" << std::endl;
    }
  } catch (const std::exception& error) {
    std::cerr << "Frozen integral probe: " << error.what() << std::endl;
    MPI_Abort(context.global, 1);
    status = 1;
  }
  MPI_Finalize();
  return status;
}
