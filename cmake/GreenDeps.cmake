
function(add_green_dependency TARGET)
    Include(FetchContent)

    FetchContent_Declare(
            ${TARGET}
            GIT_REPOSITORY https://github.com/Green-Phys/${TARGET}.git
            GIT_TAG ${GREEN_RELEASE} # or a later release
            CMAKE_ARGS -DGREEN_RELEASE=${GREEN_RELEASE}
    )

    FetchContent_MakeAvailable(${TARGET})
endfunction()