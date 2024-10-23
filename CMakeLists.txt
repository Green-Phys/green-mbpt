cmake_minimum_required(VERSION 3.5)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
project(MPI_READ_LOCK)

if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR})
    message(FATAL_ERROR "In-source builds are not allowed")
endif ()

set(CMAKE_CXX_STANDARD 11)
SET( CMAKE_CXX_FLAGS "-O3 -g -fsanitize=address")

execute_process(
        COMMAND git log -1 --format=%h
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        OUTPUT_VARIABLE TMP_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(GIT_HASH "${PROJECT_NAME}:${TMP_HASH}" CACHE INTERNAL "")


list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(Eigen3 REQUIRED)
find_package(BLAS)
find_package(MPI)
if (BLAS_FOUND)
    if (BLAS_LIBRARIES MATCHES ".*Accelerate")
        message(INFO " Using Apple BLAS")
    endif ()
    if (BLAS_LIBRARIES MATCHES ".*openblas")
        message(INFO " Using OpenBLAS")
        add_compile_definitions(USE_GEMM3M)
    endif ()
    add_compile_definitions(EIGEN_USE_BLAS)
endif ()

#manual location of green-mbpt src files
include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(src)

add_executable(mpi_read_lock.exe mpi_read_lock.cpp)
target_link_libraries(mpi_read_lock.exe PUBLIC mpi_read_lock)
target_link_libraries(mpi_read_lock.exe PUBLIC BLAS::BLAS)
target_link_libraries(mpi_read_lock.exe PUBLIC MPI::MPI_CXX)

target_compile_definitions(mpi_read_lock PRIVATE GIT_HASHES="${GIT_HASH}")

option(Build_Tests "Build tests" ON)
if (Build_Tests)
    enable_testing()
    add_subdirectory(test)
endif ()
install(TARGETS mpi_read_lock.exe DESTINATION bin)
