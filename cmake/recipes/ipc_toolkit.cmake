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
    GIT_TAG d54efbf97de0870fda8d8b44bece8a8a9d253fd2
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(ipc_toolkit)
