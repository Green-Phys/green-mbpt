project(mbpt-lib CXX)

# Weak coupling many-body framework
add_library(mbpt dyson.cpp hf_solver.cpp gw_solver.cpp gf2_solver.cpp gf2_solver_t_ewald_correction.cpp
        mbpt_q0_utils_t.cpp
        hf_cpu_kernels.cpp
        gw_cpu_kernel.cpp)
target_include_directories(mbpt PUBLIC .)
target_link_libraries(mbpt PUBLIC GREEN::PARAMS GREEN::H5PP GREEN::UTILS GREEN::NDARRAY GREEN::GRIDS GREEN::SYMMETRY GREEN::SC GREEN::IMPURITY)

# Embedding framework
add_library(embedding seet_solver.cpp)
target_include_directories(embedding PUBLIC .)
target_link_libraries(embedding PUBLIC mbpt)

add_library(transform transform.cpp)
target_include_directories(transform PUBLIC .)
target_link_libraries(transform PUBLIC GREEN::PARAMS GREEN::H5PP GREEN::UTILS GREEN::NDARRAY GREEN::GRIDS GREEN::SYMMETRY GREEN::SC)