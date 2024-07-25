
function(add_green_dependency TARGET)
    Include(FetchContent)

    set(extra_args ${ARGN})
    list(LENGTH extra_args extra_count)
    if (${extra_count} GREATER 0)
        list(GET extra_args 0 source)
        FetchContent_Declare(
            ${TARGET}
            GIT_REPOSITORY ${source}
            GIT_TAG ${GREEN_RELEASE} # or a later release
        )
    else()
        FetchContent_Declare(
            ${TARGET}
            GIT_REPOSITORY https://github.com/Green-Phys/${TARGET}.git
            GIT_TAG ${GREEN_RELEASE} # or a later release
        )
    endif()

    FetchContent_MakeAvailable(${TARGET})
endfunction()