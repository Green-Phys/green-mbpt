project(_test)

Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0 # or a later release_
)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

add_executable(solvers_test solvers_test.cpp)
target_compile_definitions(solvers_test PRIVATE TEST_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data"
        GRID_PATH="${grids_SOURCE_DIR}/data")
target_link_libraries(solvers_test
        PRIVATE
        Catch2::Catch2
        GREEN::MBPT GREEN::SC BLAS::BLAS transform)

add_executable(decomp_test decomp_test.cpp)
target_compile_definitions(decomp_test PRIVATE TEST_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data"
        GRID_PATH="${grids_SOURCE_DIR}/data")
target_link_libraries(decomp_test
        PRIVATE
        Catch2::Catch2WithMain
        GREEN::MBPT GREEN::SC BLAS::BLAS transform)

include(CTest)
include(Catch)
catch_discover_tests(solvers_test)
catch_discover_tests(decomp_test)
