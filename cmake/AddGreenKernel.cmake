
function(add_green_kernel CUSTOM_KERNELS_IN)
    set(CUSTOM_KERNELS_TMP "${CUSTOM_KERNELS_IN}")
    set(CUSTOM_KERNELS_LST "")
    foreach(KERNEL ${CUSTOM_KERNELS_TMP})
        string(REGEX MATCH "([^/]+)/?$" KERNEL_NAME ${KERNEL})
        message("Adding kernel ${KERNEL_NAME} ${KERNEL}")
        if(NOT DEFINED KERNEL_NAME)
            message(FATAL_ERROR "Can not extract kernel name")
        endif()

        Include(FetchContent)

        FetchContent_Declare(
            ${KERNEL_NAME}
            GIT_REPOSITORY ${KERNEL}
            GIT_TAG ${GREEN_RELEASE} # or a later release
        )

        FetchContent_MakeAvailable(${KERNEL_NAME})
        list(APPEND CUSTOM_KERNELS_LST ${KERNEL})
    endforeach()
    set(CUSTOM_KERNELS "${CUSTOM_KERNELS_LST}" PARENT_SCOPE)
endfunction()