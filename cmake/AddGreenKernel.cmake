
function(add_green_kernel CUSTOM_KERNEL KERNEL_URL)
    message("Adding kernel ${CUSTOM_KERNEL} ${KERNEL_URL}")
    Include(FetchContent)

    FetchContent_Declare(
            ${CUSTOM_KERNEL}
            GIT_REPOSITORY ${KERNEL_URL}
            GIT_TAG ${GREEN_RELEASE} # or a later release
    )

    FetchContent_MakeAvailable(${CUSTOM_KERNEL})
endfunction()