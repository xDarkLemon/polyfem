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
    GIT_TAG fe05427cd799a746f4cc01caf8fadbe6977323f5
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(polysolve)
