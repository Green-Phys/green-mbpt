include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz
        URL_HASH MD5=7e11f6cfcf6498324ac82d567dcb891e
        DOWNLOAD_EXTRACT_TIMESTAMP true
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

include(GoogleTest)

set(test_src
  test_main.cpp
  test_mpi.cpp
  test_buffer.cpp
  test_shared_memory_region.cpp
)

add_executable(runtests
        ${test_src}
        )
target_link_libraries(runtests PUBLIC gtest)
target_link_libraries(runtests PRIVATE mpi_read_lock)
target_link_libraries(runtests PUBLIC BLAS::BLAS)
target_link_libraries(runtests PUBLIC MPI::MPI_CXX)
target_compile_definitions(runtests PRIVATE HDF5_DATA_DIR="${CMAKE_CURRENT_LIST_DIR}/../hdf5/")

if (TEST_XML_OUTPUT)
    add_test(NAME all_tests
             COMMAND runtests --gtest_output="xml:report.xml")
else ()
    gtest_discover_tests(runtests)
endif ()

