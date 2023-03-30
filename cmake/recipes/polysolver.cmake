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
    GIT_TAG 5ac5e5602e0d910159c9a1549c388b873edcb660
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(polysolve)
