project(mbpt-lib CXX)

add_library(buffered_reader buffer.cpp reader.cpp chunk_reader.cpp)
target_link_libraries(buffered_reader PUBLIC Eigen3::Eigen MPI::MPI_CXX hdf5::hdf5)
target_include_directories(buffered_reader PUBLIC ${CMAKE_SOURCE_DIR}/src/green/mbpt/buffered_reader)

add_library(mbpt dyson.cpp hf_solver.cpp gw_solver.cpp gf2_solver.cpp gf2_solver_t_ewald_correction.cpp mbpt_q0_utils_t.cpp
        hf_cpu_kernels.cpp
        gw_cpu_kernel.cpp)
target_include_directories(mbpt PUBLIC .)
target_link_libraries(mbpt PUBLIC GREEN::PARAMS GREEN::H5PP GREEN::UTILS GREEN::NDARRAY GREEN::GRIDS GREEN::SYMMETRY GREEN::SC buffered_reader)
