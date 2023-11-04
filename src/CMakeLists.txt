project(mbpt-lib CXX)

add_library(mbpt dyson.cpp hf_solver.cpp gw_solver.cpp gw_solver_prec.cpp gf2_solver.cpp gf2_solver_t_ewald_correction.cpp)
target_include_directories(mbpt PUBLIC .)
target_link_libraries(mbpt PUBLIC GREEN::PARAMS GREEN::H5PP GREEN::UTILS GREEN::NDARRAY GREEN::GRIDS GREEN::SYMMETRY)