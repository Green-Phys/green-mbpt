cmake_minimum_required(VERSION 3.18)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(GREEN_SC_SRC "/Users/gharsha/Documents/Codes/green-sc")
project(mbpt)

if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR})
    message(FATAL_ERROR "In-source builds are not allowed")
endif ()

################ GREEN RELEASE VERSION #####################
set(GREEN_RELEASE origin/main)

set(CMAKE_CXX_STANDARD 17)

execute_process(
        COMMAND git log -1 --format=%h
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        OUTPUT_VARIABLE TMP_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(GIT_HASH "${PROJECT_NAME}:${TMP_HASH}" CACHE INTERNAL "")


list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(BLAS)
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

include(GreenDeps)

add_green_dependency(green-grids)
add_green_dependency(green-h5pp)
add_green_dependency(green-ndarray)
add_green_dependency(green-params)
add_green_dependency(green-sc)
# set(CUSTOM_PATH "/Users/gharsha/Documents/Codes/green-sc")
# add_green_dependency(green-sc ${CUSTOM_PATH})
add_green_dependency(green-symmetry)
add_green_dependency(green-utils)

if (CUSTOM_KERNEL)
    include(AddGreenKernel)
    add_green_kernel(${CUSTOM_KERNEL} ${GREEN_KERNEL_URL})
endif ()

add_subdirectory(src)
add_library(GREEN::MBPT ALIAS mbpt)
if (CUSTOM_KERNEL)
    if (NOT DEFINED GREEN_CUSTOM_KERNEL_LIB)
        message(FATAL_ERROR "You need to define a cmake library for custom kernel!")
    endif ()
    target_link_libraries(mbpt PUBLIC ${GREEN_CUSTOM_KERNEL_LIB})
    target_compile_definitions(mbpt PUBLIC CUSTOM_KERNEL)
    if (NOT DEFINED GREEN_CUSTOM_KERNEL_ENUM)
        message(FATAL_ERROR "You need to define a enum for custom kernel type!")
    endif ()
    target_compile_definitions(mbpt PUBLIC GREEN_CUSTOM_KERNEL_ENUM=${GREEN_CUSTOM_KERNEL_ENUM})
    if (GREEN_CUSTOM_KERNEL_HEADER)
        target_compile_definitions(mbpt PUBLIC GREEN_CUSTOM_KERNEL_HEADER=${GREEN_CUSTOM_KERNEL_HEADER})
    endif ()
endif ()

add_executable(mbpt.exe main.cpp)
target_link_libraries(mbpt.exe PUBLIC GREEN::MBPT)
if (BLAS_FOUND)
    target_link_libraries(mbpt.exe PUBLIC BLAS::BLAS)
endif (BLAS_FOUND)
target_compile_definitions(mbpt.exe PRIVATE GIT_HASHES="${GIT_HASH}")

option(Build_Tests "Build tests" ON)
if (Build_Tests)
    enable_testing()
    add_subdirectory(test)
endif ()
install(TARGETS mbpt.exe DESTINATION bin)
install(DIRECTORY python DESTINATION . FILES_MATCHING PATTERN "*.py")