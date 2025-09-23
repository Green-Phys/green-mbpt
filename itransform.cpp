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

#include <green/params/params.h>
#include <green/utils/mpi_shared.h>
#include <green/transform/transform.h>

#include <iostream>

#include "mpi.h"

green::transform::int_transform parse_input(int argc, char** argv) {
  std::string name = R"(
▀█▀ █▀▀▄ ▀▀█▀▀ █▀▀ █▀▀▀ █▀▀█ █▀▀█ █
 █  █  █   █   █▀▀ █ ▀█ █▄▄▀ █▄▄█ █
▄█▄ ▀  ▀   ▀   ▀▀▀ ▀▀▀▀ ▀ ▀▀ ▀  ▀ ▀▀▀

    ▀▀█▀▀ █▀▀█ █▀▀█ █▀▀▄ █▀▀ █▀▀ █▀▀█ █▀▀█ █▀▄▀█ █▀▀ █▀▀█
      █   █▄▄▀ █▄▄█ █  █ ▀▀█ █▀▀ █  █ █▄▄▀ █ ▀ █ █▀▀ █▄▄▀
      █   ▀ ▀▀ ▀  ▀ ▀  ▀ ▀▀▀ ▀   ▀▀▀▀ ▀ ▀▀ ▀   ▀ ▀▀▀ ▀ ▀▀)";
  green::params::params p(name);
  p.define<std::string>("input_file", "Path to input  HDF5 file with transformations", "transform.h5");
  p.define<std::string>("in_file", "Path to input HDF5 file with Weak Coupling input", "input.h5");
  p.define<std::string>("in_int_file", "Path to input HDF5 files with Coulomb integrals", "df_int");
  p.define<std::string>("dc_int_path", "Name for double counting integral-files", "dc_int");
  p.define<int>("verbose", "Verbosity level", 0);
  p.define<bool>("transform", "Evaluate transformed three-center integrals", false);
  p.parse(argc, argv);
  if (!p.parse(argc, argv)) {
    if (!green::utils::context.global_rank) p.help();
    MPI_Finalize();
    std::exit(-1);
  }
  if (!green::utils::context.global_rank) p.print();

  return green::transform::int_transform{p["input_file"], p["in_file"], p["in_int_file"], p["input_file"], p["dc_int_path"], static_cast<int>(p["transform"].as<bool>()), p["verbose"]};
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  green::transform::int_transform params = parse_input(argc, argv);

  green::transform::int_transformer transformer(params);
  transformer.transform_3point();

  MPI_Finalize();
}
