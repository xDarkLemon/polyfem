# Polyfem Solvers
# License: MIT

if(TARGET polysolve)
    return()
endif()

message(STATUS "Third-party: creating target 'polysolve'")

include(FetchContent)
FetchContent_Declare(
    polysolve
    GIT_REPOSITORY https://github.com/AlexTru96/polysolve.git
    GIT_TAG 288b9bda62af8176c9fdbc926021a10a696d271d
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(polysolve)
