
function(add_green_dependency TARGET)
    Include(FetchContent)

    FetchContent_Declare(
            ${TARGET}
            GIT_REPOSITORY https://github.com/Green-Phys/${TARGET}.git
            GIT_TAG origin/main # or a later release
    )

    FetchContent_MakeAvailable(${TARGET})
endfunction()