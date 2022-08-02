# IPC Toolkit (https://github.com/ipc-sim/ipc-toolkit)
# License: MIT

if(TARGET ipc::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'ipc::toolkit'")

include(FetchContent)
FetchContent_Declare(
    ipc_toolkit
    GIT_REPOSITORY https://github.com/AlexTru96/ipc-toolkit
    GIT_TAG c5afdccdcc2637af6351072cb883e60b30f6b005
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(ipc_toolkit)
