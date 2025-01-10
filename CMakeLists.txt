cmake_minimum_required(VERSION 3.18)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
project(mbpt)

if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR})
    message(FATAL_ERROR "In-source builds are not allowed")
endif ()

################ GREEN RELEASE VERSION #####################
set(GREEN_RELEASE origin/main)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-g -fno-omit-frame-pointer")

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
add_green_dependency(green-symmetry)
add_green_dependency(green-utils)
add_green_dependency(green-integrals)

if (CUSTOM_KERNELS)
    include(AddGreenKernel)
    add_green_kernel("${CUSTOM_KERNELS}")
endif ()

add_subdirectory(src)
add_library(GREEN::MBPT ALIAS mbpt)
if (CUSTOM_KERNELS)
    list(LENGTH CUSTOM_KERNELS NUM_OF_KERNELS)
    list(LENGTH CUSTOM_KERNEL_ENUMS NUM_OF_ENUMS)
    list(LENGTH CUSTOM_KERNEL_HEADERS NUM_OF_HEADERS)
    list(LENGTH CUSTOM_KERNEL_NAMESPACES NUM_OF_NS)
    list(LENGTH CUSTOM_KERNEL_LIBS NUM_OF_LIBS)
    if( NOT ( ( ${NUM_OF_LIBS} EQUAL ${NUM_OF_ENUMS} ) 
          AND ( ${NUM_OF_LIBS} EQUAL ${NUM_OF_NS} )
          AND ( ${NUM_OF_LIBS} EQUAL ${NUM_OF_HEADERS} ) ) )
        message(FATAL_ERROR "Inconsistent number of enums, libs, namespaces or headers!!!")
    endif()
    foreach(KERNEL_NUM RANGE ${NUM_OF_KERNELS})
        math(EXPR BASE_1 "${KERNEL_NUM} + 1")
        list(GET CUSTOM_KERNEL_HEADERS ${KERNEL_NUM} HEADER)
        list(GET CUSTOM_KERNEL_ENUMS ${KERNEL_NUM} ENUM)
        list(GET CUSTOM_KERNEL_NAMESPACES ${KERNEL_NUM} NS)
        list(GET CUSTOM_KERNEL_LIBS ${KERNEL_NUM} LIB)
        if( "${ENUM}" STREQUAL "" )
            break()
        endif()
        message(STATUS "KERNEL No. ${BASE_1}: ${HEADER} ${ENUM} ${NS} ${LIB}")
        target_compile_definitions(mbpt PUBLIC GREEN_CUSTOM_KERNEL_HEADER_${KERNEL_NUM}=${HEADER})
        target_compile_definitions(mbpt PUBLIC GREEN_CUSTOM_KERNEL_ENUM_${KERNEL_NUM}=${ENUM})
        target_compile_definitions(mbpt PUBLIC GREEN_CUSTOM_KERNEL_NS_${KERNEL_NUM}=${NS})
        target_link_libraries(mbpt PUBLIC ${LIB})
    endforeach()
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
