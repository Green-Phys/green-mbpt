include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

include(GoogleTest)

set(test_src
  test_buffer.cpp
)

add_executable(runtests
        ${test_src}
        )
target_link_libraries(runtests PUBLIC gtest_main)
target_link_libraries(runtests PRIVATE mpi_read_lock)
target_link_libraries(runtests PUBLIC BLAS::BLAS)
target_link_libraries(runtests PUBLIC MPI::MPI_CXX)
target_compile_definitions(runtests PRIVATE MATHEMATICA_DATA_DIR="${CMAKE_CURRENT_LIST_DIR}/../Mathematica/")

if (TEST_XML_OUTPUT)
    add_test(NAME all_tests
             COMMAND runtests --gtest_output="xml:report.xml")
else ()
    gtest_discover_tests(runtests)
endif ()

