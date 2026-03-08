
function(add_green_dependency TARGET VERSION)
    Include(FetchContent)

    FetchContent_Declare(
            ${TARGET}
            GIT_REPOSITORY https://github.com/Green-Phys/${TARGET}.git
            GIT_TAG ${VERSION}
    )

    FetchContent_MakeAvailable(${TARGET})
endfunction()