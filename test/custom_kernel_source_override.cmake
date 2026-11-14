# Configure-only regression: both URL spellings must use the explicit source
# checkout. Fully disconnected mode prevents any fallback network fetch.
if(NOT DEFINED KERNEL_HELPER OR NOT DEFINED TEST_BINARY_ROOT)
    message(FATAL_ERROR "KERNEL_HELPER and TEST_BINARY_ROOT are required")
endif()

foreach(suffix plain dotgit)
    set(project_dir "${TEST_BINARY_ROOT}/custom-kernel-override-${suffix}")
    set(fixture_dir "${project_dir}/fixture")
    file(MAKE_DIRECTORY "${fixture_dir}")
    file(WRITE "${fixture_dir}/CMakeLists.txt"
        "set(TEST_KERNEL_SOURCE_OVERRIDE_USED TRUE CACHE INTERNAL \"\" FORCE)\n")
    set(url "https://example.invalid/green-gpu")
    if(suffix STREQUAL "dotgit")
        string(APPEND url ".git")
    endif()
    file(WRITE "${project_dir}/CMakeLists.txt"
        "cmake_minimum_required(VERSION 3.18)\n"
        "project(kernel_override LANGUAGES NONE)\n"
        "set(GREEN_RELEASE unused)\n"
        "set(FETCHCONTENT_FULLY_DISCONNECTED ON)\n"
        "set(TEST_KERNEL_SOURCE_OVERRIDE_USED FALSE CACHE INTERNAL \"\" FORCE)\n"
        "set(FETCHCONTENT_SOURCE_DIR_GREEN-GPU \"${fixture_dir}\")\n"
        "include(\"${KERNEL_HELPER}\")\n"
        "add_green_kernel(\"${url}\")\n"
        "if(NOT TEST_KERNEL_SOURCE_OVERRIDE_USED)\n"
        "  message(FATAL_ERROR \"Explicit GPU checkout was not selected\")\n"
        "endif()\n")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -S "${project_dir}" -B "${project_dir}/build"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "${suffix} URL ignored the source override:\n${output}\n${error}")
    endif()
endforeach()
