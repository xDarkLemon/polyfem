if(TARGET Thrust::Thrust)
    return()
endif()

    message(STATUS "Third-party: creating target 'Thrust::Thrust'")

    include(FetchContent)
    FetchContent_Declare(
        Thrust
        GIT_REPOSITORY https://github.com/NVIDIA/thrust
        GIT_TAG tags/1.17.0
        GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(Thrust)
    if(NOT Thrust_POPULATED)
        FetchContent_Populate(Thrust)
    endif()
    set(THRUST_INCLUDE_DIRS ${Thrust_SOURCE_DIR})

    install(DIRECTORY ${THRUST_INCLUDE_DIRS}/Thrust
        DESTINATION include
    )

add_library(Thrust_CUDA INTERFACE)
add_library(Thrust::Thrust ALIAS Thrust_CUDA)

include(GNUInstallDirs)
target_include_directories(Thrust_CUDA SYSTEM INTERFACE
    $<BUILD_INTERFACE:${THRUST_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Install rules
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME Thrust)
set_target_properties(Thrust_CUDA PROPERTIES EXPORT_NAME Thrust)
install(DIRECTORY ${THRUST_INCLUDE_DIRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS Thrust_CUDA EXPORT Thrust_Targets)
install(EXPORT Thrust_Targets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/thrust NAMESPACE Thrust::)