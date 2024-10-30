Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0 # or a later release_
)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

enable_testing()

set(test_src
  test_main.cpp
  test_mpi.cpp
  test_access_counter.cpp
  test_buffer.cpp
  test_shared_memory_region.cpp
  test_chunk_reader.cpp
  test_bufferedSI.cpp
  test_age_out_buffer.cpp
  test_hash.cpp
)

add_executable(runtests
        ${test_src}
        )

target_compile_definitions(runtests PRIVATE TEST_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data"
        GRID_PATH="${grids_SOURCE_DIR}/data")
target_link_libraries(runtests
        PRIVATE
        Catch2::Catch2)
target_link_libraries(runtests PRIVATE mpi_read_lock)
target_link_libraries(runtests PUBLIC BLAS::BLAS)
target_link_libraries(runtests PUBLIC MPI::MPI_CXX)
target_link_libraries(runtests PUBLIC OpenSSL::Crypto)
target_compile_definitions(runtests PRIVATE HDF5_DATA_DIR="${CMAKE_CURRENT_LIST_DIR}/../hdf5/")


include(CTest)
include(Catch)

