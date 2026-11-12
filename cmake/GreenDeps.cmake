function(add_green_dependency TARGET)
    Include(FetchContent)

    set(extra_args ${ARGN})
    list(LENGTH extra_args extra_count)
    if (${extra_count} GREATER 0)
        list(GET extra_args 0 source)
        list(GET extra_args 1 custom_release)
        FetchContent_Declare(
            ${TARGET}
            GIT_REPOSITORY ${source}
            GIT_TAG ${custom_release} # or a later release
            CMAKE_ARGS -DGREEN_RELEASE=${GREEN_RELEASE}
        )
    else()
        set(dependency_repository https://github.com/Green-Phys/${TARGET}.git)
        set(dependency_revision ${GREEN_RELEASE})
        if("${TARGET}" STREQUAL "green-symmetry")
            set(GREEN_SYMMETRY_GIT_REPOSITORY "https://github.com/Green-Phys/green-symmetry.git" CACHE STRING "SG reader dependency repository")
            set(GREEN_SYMMETRY_GIT_TAG "${GREEN_RELEASE}" CACHE STRING "Revision containing the SG reader API")
            set(dependency_repository ${GREEN_SYMMETRY_GIT_REPOSITORY})
            set(dependency_revision ${GREEN_SYMMETRY_GIT_TAG})
        endif()
        FetchContent_Declare(
            ${TARGET}
            GIT_REPOSITORY ${dependency_repository}
            GIT_TAG ${dependency_revision}
            CMAKE_ARGS -DGREEN_RELEASE=${GREEN_RELEASE}
        )
    endif()

    FetchContent_MakeAvailable(${TARGET})
endfunction()
