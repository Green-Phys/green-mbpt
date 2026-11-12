
function(add_green_kernel CUSTOM_KERNELS_IN)
    set(CUSTOM_KERNELS_TMP "${CUSTOM_KERNELS_IN}")
    set(CUSTOM_KERNELS_LST "")
    foreach(KERNEL ${CUSTOM_KERNELS_TMP})
        string(REGEX MATCH "([^/]+)/?$" KERNEL_NAME ${KERNEL})
        string(REGEX REPLACE "\\.git$" "" KERNEL_NAME "${KERNEL_NAME}")
        message("Adding kernel ${KERNEL_NAME} ${KERNEL}")
        if(NOT DEFINED KERNEL_NAME)
            message(FATAL_ERROR "Can not extract kernel name")
        endif()

        Include(FetchContent)
        set(kernel_revision ${GREEN_RELEASE})
        set(kernel_repository ${KERNEL})
        if("${KERNEL_NAME}" STREQUAL "green-gpu")
            set(GREEN_GPU_GIT_REPOSITORY "${KERNEL}" CACHE STRING "Repository containing the SG GPU reader")
            set(GREEN_GPU_GIT_TAG "${GREEN_RELEASE}" CACHE STRING "Revision containing SG GPU acquisition support")
            set(kernel_repository ${GREEN_GPU_GIT_REPOSITORY})
            set(kernel_revision ${GREEN_GPU_GIT_TAG})
        endif()

        FetchContent_Declare(
            ${KERNEL_NAME}
            GIT_REPOSITORY ${kernel_repository}
            GIT_TAG ${kernel_revision}
        )

        FetchContent_MakeAvailable(${KERNEL_NAME})
        list(APPEND CUSTOM_KERNELS_LST ${KERNEL})
    endforeach()
    set(CUSTOM_KERNELS "${CUSTOM_KERNELS_LST}" PARENT_SCOPE)
endfunction()
