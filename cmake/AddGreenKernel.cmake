
function(add_green_kernel CUSTOM_KERNELS_IN)
    set(CUSTOM_KERNELS_TMP "${CUSTOM_KERNELS_IN}")
    set(CUSTOM_KERNELS_LST "")
    foreach(KERNEL_SPEC ${CUSTOM_KERNELS_TMP})
        # Supported forms:
        #   - kernel-name
        #   - kernel-name=tag
        #   - https://.../kernel-name(.git)
        #   - https://.../kernel-name(.git)=tag
        # If no tag is provided, use GREEN_RELEASE.
        set(KERNEL_ID "${KERNEL_SPEC}")
        set(KERNEL_TAG "${GREEN_RELEASE}")

        # Parse optional "kernel=tag" syntax.
        # ^ and $ force full-string match, and each side of '=' must be non-empty.
        # Captures: CMAKE_MATCH_1 -> kernel id/repo, CMAKE_MATCH_2 -> tag/version.
        if(KERNEL_SPEC MATCHES "^([^=]+)=([^=]+)$")
            set(KERNEL_ID "${CMAKE_MATCH_1}")
            set(KERNEL_TAG "${CMAKE_MATCH_2}")
        endif()

        # Parse optional "https://.../kernel(.git)" syntax. If it matches,
        # use the full URL as repo and extract kernel name from it. Otherwise,
        # treat kernel id as kernel name and construct repo URL using Green-Phys convention.
        if(KERNEL_ID MATCHES "^https?://")
            set(KERNEL_REPO "${KERNEL_ID}")
            string(REGEX MATCH "([^/]+?)(\\.git)?/?$" _ "${KERNEL_ID}")
            set(KERNEL_NAME "${CMAKE_MATCH_1}")
        else()
            set(KERNEL_NAME "${KERNEL_ID}")
            set(KERNEL_REPO "https://github.com/Green-Phys/${KERNEL_NAME}.git")
        endif()

        message(STATUS "Adding kernel ${KERNEL_NAME} from ${KERNEL_REPO} @ ${KERNEL_TAG}")
        if(NOT DEFINED KERNEL_NAME OR "${KERNEL_NAME}" STREQUAL "")
            message(FATAL_ERROR "Can not extract kernel name")
        endif()

        Include(FetchContent)

        FetchContent_Declare(
            ${KERNEL_NAME}
            GIT_REPOSITORY ${KERNEL_REPO}
            GIT_TAG ${KERNEL_TAG}
        )

        FetchContent_MakeAvailable(${KERNEL_NAME})
        list(APPEND CUSTOM_KERNELS_LST ${KERNEL_SPEC})
    endforeach()
    set(CUSTOM_KERNELS "${CUSTOM_KERNELS_LST}" PARENT_SCOPE)
endfunction()
