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
    GIT_TAG 803de42b475165177ebfb5bfa59d7e08ddc7a61d
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(polysolve)
