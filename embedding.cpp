/*
 * Copyright (c) 2023 University of Michigan
 *
 */

#include <green/mbpt/dyson.h>
#include <green/mbpt/mbpt_run.h>
#include <green/embedding/embedding_run.h>
#include <green/sc/sc_loop.h>
#include <mpi.h>

template <typename A>
auto init_solver(const green::params::params p) {}

int  main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  green::utils::context;
  std::string hashes = std::string(GIT_HASHES);
  int         pos;
  while ((pos = hashes.find("**")) != std::string::npos) hashes.replace(pos, 2, "\n");
  std::string name = R"(
 █▀▀█ █▀▀█ █▀▀ █▀▀ █▀▀▄
 █ ▄▄ █▄▄▀ █▀▀ █▀▀ █  █
 █▄▄█ ▀ ▀▀ ▀▀▀ ▀▀▀ ▀  ▀

 █▀▀▀ █▀▄▀█ █▀▀▄ █▀▀ █▀▀▄ █▀▀▄  ▀  █▀▀▄ █▀▀▀
 █▀▀▀ █ ▀ █ █▀▀▄ █▀▀ █  █ █  █ ▀█▀ █  █ █ ▀█
 █▄▄▄ ▀   ▀ ▀▀▀  ▀▀▀ ▀▀▀  ▀▀▀  ▀▀▀ ▀  ▀ ▀▀▀▀)";

  auto        p    = green::params::params(
      name + "\n\nMichigan Quantum Embedding solver.\n\n=====================================\nGit hashes:\n" +
      hashes + "\n=====================================");
  green::sc::define_parameters(p);
  green::symmetry::define_parameters(p);
  green::grids::define_parameters(p);
  green::mbpt::define_parameters(p);
  green::embedding::define_parameters(p);

  if (!p.parse(argc, argv)) {
    if (!green::utils::context.global_rank) p.help();
    MPI_Finalize();
    return -1;
  }
  if (!green::utils::context.global_rank) p.print();

  try {
    green::mbpt::check_input(p);
    green::sc::sc_loop<green::mbpt::shared_mem_dyson> sc(MPI_COMM_WORLD, p);
    green::embedding::run(sc, p);
  } catch (std::exception& e) {
    std::cerr << "Error on node " << green::utils::context.global_rank << ": " << e.what() << std::endl;
    MPI_Abort(green::utils::context.global, -1);
  }

  MPI_Finalize();
  return 0;
}