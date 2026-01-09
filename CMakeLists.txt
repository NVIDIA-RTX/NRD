cmake_minimum_required(VERSION 3.22...3.30)

include(CMakeDependentOption)
include(FetchContent)

# Architecture
if((CMAKE_GENERATOR_PLATFORM MATCHES "ARM64") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64"))
    set(IS_ARM64 TRUE)
else()
    set(IS_ARM64 FALSE)
endif()

# Options
option(NRD_STATIC_LIBRARY "Build static library" OFF)
option(NRD_NRI "Pull, build and include NRI into NRD SDK package" OFF)
option(NRD_SUPPORTS_VIEWPORT_OFFSET "Enable 'CommonSettings::rectOrigin' support" OFF)
option(NRD_SUPPORTS_CHECKERBOARD "Enable 'checkerboardMode' support" ON)
option(NRD_SUPPORTS_HISTORY_CONFIDENCE "Enable 'IN_DIFF_CONFIDENCE' and 'IN_SPEC_CONFIDENCE' support" ON)
option(NRD_SUPPORTS_DISOCCLUSION_THRESHOLD_MIX "Enable 'IN_DISOCCLUSION_THRESHOLD_MIX' support" ON)
option(NRD_SUPPORTS_BASECOLOR_METALNESS "Enable 'IN_BASECOLOR_METALNESS' support" ON)
option(NRD_SUPPORTS_ANTIFIREFLY "Enable 'enableAntiFirefly' support" ON)
option(NRD_EMBEDS_SPIRV_SHADERS "NRD embeds SPIRV shaders" ON)
option(REBLUR_PERFORMANCE_MODE "Better performance and worse image quality, can be useful for consoles" OFF)

cmake_dependent_option(NRD_EMBEDS_DXIL_SHADERS "NRD embeds DXIL shaders" ON "WIN32" OFF)
cmake_dependent_option(NRD_EMBEDS_DXBC_SHADERS "NRD embeds DXBC shaders" ON "WIN32" OFF)

set(NRD_SHADERS_PATH "" CACHE STRING "Shader output path override")
set(NRD_NORMAL_ENCODING "2" CACHE STRING "Normal encoding variant (0-4, matches nrd::NormalEncoding)")
set(NRD_ROUGHNESS_ENCODING "1" CACHE STRING "Roughness encoding variant (0-2, matches nrd::RoughnessEncoding)")

# Self-build?
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR}) # if not submodule
    get_property(IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

    if(IS_MULTI_CONFIG)
        set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "")
    endif()

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/_Bin" CACHE STRING "") # .dll
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/_Bin" CACHE STRING "") # .so
endif()

# Create project
file(READ "Include/NRD.h" ver_h)
string(REGEX MATCH "NRD_VERSION_MAJOR ([0-9]*)" _ ${ver_h})
set(NRD_VERSION_MAJOR ${CMAKE_MATCH_1})
string(REGEX MATCH "NRD_VERSION_MINOR ([0-9]*)" _ ${ver_h})
set(NRD_VERSION_MINOR ${CMAKE_MATCH_1})
string(REGEX MATCH "NRD_VERSION_BUILD ([0-9]*)" _ ${ver_h})
set(NRD_VERSION_BUILD ${CMAKE_MATCH_1})
message("NRD: v${NRD_VERSION_MAJOR}.${NRD_VERSION_MINOR}.${NRD_VERSION_BUILD}")

project(NRD VERSION ${NRD_VERSION_MAJOR}.${NRD_VERSION_MINOR}.${NRD_VERSION_BUILD} LANGUAGES CXX)

# Enable grouping of source files into folders in IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Generate "NRDConfig.hlsli"
function(to_int_bool VAR_NAME INPUT_VAR_NAME)
    # Convert "bool" to "0/1"
    if (${${INPUT_VAR_NAME}})
        set(${VAR_NAME} "1" PARENT_SCOPE)
    else()
        set(${VAR_NAME} "0" PARENT_SCOPE)
    endif()
endfunction()

to_int_bool(VIEWPORT_OFFSET_INT NRD_SUPPORTS_VIEWPORT_OFFSET)
to_int_bool(CHECKERBOARD_INT NRD_SUPPORTS_CHECKERBOARD)
to_int_bool(HISTORY_CONFIDENCE_INT NRD_SUPPORTS_HISTORY_CONFIDENCE)
to_int_bool(DISOCCLUSION_THRESHOLD_MIX_INT NRD_SUPPORTS_DISOCCLUSION_THRESHOLD_MIX)
to_int_bool(BASECOLOR_METALNESS_INT NRD_SUPPORTS_BASECOLOR_METALNESS)
to_int_bool(ANTIFIREFLY_INT NRD_SUPPORTS_ANTIFIREFLY)
to_int_bool(REBLUR_PERFORMANCE_MODE_INT REBLUR_PERFORMANCE_MODE)

file(WRITE "Shaders/Include/NRDConfig.hlsli"
    "// This file is auto-generated during project deployment. Do not modify!\n"
    "#define NRD_NORMAL_ENCODING ${NRD_NORMAL_ENCODING}\n"
    "#define NRD_ROUGHNESS_ENCODING ${NRD_ROUGHNESS_ENCODING}\n"
    "#define NRD_SUPPORTS_VIEWPORT_OFFSET ${VIEWPORT_OFFSET_INT}\n"
    "#define NRD_SUPPORTS_CHECKERBOARD ${CHECKERBOARD_INT}\n"
    "#define NRD_SUPPORTS_HISTORY_CONFIDENCE ${HISTORY_CONFIDENCE_INT}\n"
    "#define NRD_SUPPORTS_DISOCCLUSION_THRESHOLD_MIX ${DISOCCLUSION_THRESHOLD_MIX_INT}\n"
    "#define NRD_SUPPORTS_BASECOLOR_METALNESS ${BASECOLOR_METALNESS_INT}\n"
    "#define NRD_SUPPORTS_ANTIFIREFLY ${ANTIFIREFLY_INT}\n"
    "#define REBLUR_PERFORMANCE_MODE ${REBLUR_PERFORMANCE_MODE_INT}\n"
)

# SPIRV offsets (must respect shader resources limits)
set(SPIRV_SREG_OFFSET 0)
set(SPIRV_BREG_OFFSET 2)
set(SPIRV_UREG_OFFSET 3)
set(SPIRV_TREG_OFFSET 20)

# Compile definitions (public)
set(COMPILE_DEFINITIONS "")
foreach(opt IN ITEMS
    NRD_SUPPORTS_VIEWPORT_OFFSET
    NRD_SUPPORTS_CHECKERBOARD
    NRD_SUPPORTS_HISTORY_CONFIDENCE
    NRD_SUPPORTS_DISOCCLUSION_THRESHOLD_MIX
    NRD_SUPPORTS_BASECOLOR_METALNESS
    NRD_SUPPORTS_ANTIFIREFLY
    REBLUR_PERFORMANCE_MODE
)
    if(${opt})
        message(STATUS "${opt}")
        list(APPEND COMPILE_DEFINITIONS "${opt}=1")
    else()
        list(APPEND COMPILE_DEFINITIONS "${opt}=0")
    endif()
endforeach()

message(STATUS "NRD_NORMAL_ENCODING = ${NRD_NORMAL_ENCODING}")
message(STATUS "NRD_ROUGHNESS_ENCODING = ${NRD_ROUGHNESS_ENCODING}")

# Download dependencies
set(DEPS)

if(NOT TARGET ShaderMake)
    # ShaderMake
    option(SHADERMAKE_TOOL "" ON)

    FetchContent_Declare(
        shadermake
        DOWNLOAD_EXTRACT_TIMESTAMP 1
        DOWNLOAD_NO_PROGRESS 1
        URL https://github.com/NVIDIA-RTX/ShaderMake/archive/18f5a344e7ca8fa65daaf079d07bc8ce38453e05.zip
    )
    list(APPEND DEPS shadermake)
endif()

if(NOT TARGET MathLib)
    # MathLib
    FetchContent_Declare(
        mathlib
        DOWNLOAD_EXTRACT_TIMESTAMP 1
        DOWNLOAD_NO_PROGRESS 1
        URL https://github.com/NVIDIA-RTX/MathLib/archive/refs/tags/v10.zip
    )
    list(APPEND DEPS mathlib)
endif()

if(NOT TARGET NRI AND NRD_NRI)
    # NRI
    option(NRI_ENABLE_NVAPI "" OFF)
    option(NRI_ENABLE_AMDAGS "" OFF)
    option(NRI_ENABLE_NVTX_SUPPORT "" OFF)
    option(NRI_ENABLE_NONE_SUPPORT "" OFF)
    cmake_dependent_option(NRI_ENABLE_VK_SUPPORT "" ON "NRD_EMBEDS_SPIRV_SHADERS" OFF)
    cmake_dependent_option(NRI_ENABLE_D3D12_SUPPORT "" ON "NRD_EMBEDS_DXIL_SHADERS" OFF)
    cmake_dependent_option(NRI_ENABLE_D3D11_SUPPORT "" ON "NRD_EMBEDS_DXBC_SHADERS" OFF)

    FetchContent_Declare(
        nri
        DOWNLOAD_EXTRACT_TIMESTAMP 1
        DOWNLOAD_NO_PROGRESS 1
        URL https://github.com/NVIDIA-RTX/NRI/archive/refs/tags/v177.zip
    )
    list(APPEND DEPS nri)
endif()

if(DEPS)
    message("NRD: downloading dependencies:")
    message(STATUS "${DEPS} ...")

    FetchContent_MakeAvailable(${DEPS})
endif()

# NRD
set(GLOB_INCUDE
    "Include/NRD.h"
    "Include/NRDDescs.h"
    "Include/NRDSettings.h"
)
source_group("Include" FILES ${GLOB_INCUDE})

set(GLOB_SOURCE
    "Source/InstanceImpl.cpp"
    "Source/InstanceImpl.h"
    "Source/Reblur.cpp"
    "Source/Reference.cpp"
    "Source/Relax.cpp"
    "Source/Sigma.cpp"
    "Source/StdAllocator.h"
    "Source/Timer.cpp"
    "Source/Timer.h"
    "Source/Wrapper.cpp"
)
source_group("Source" FILES ${GLOB_SOURCE})

set(GLOB_DENOISERS
    "Source/Denoisers/Reblur_Diffuse.hpp"
    "Source/Denoisers/Reblur_DiffuseDirectionalOcclusion.hpp"
    "Source/Denoisers/Reblur_DiffuseOcclusion.hpp"
    "Source/Denoisers/Reblur_DiffuseSh.hpp"
    "Source/Denoisers/Reblur_DiffuseSpecular.hpp"
    "Source/Denoisers/Reblur_DiffuseSpecularOcclusion.hpp"
    "Source/Denoisers/Reblur_DiffuseSpecularSh.hpp"
    "Source/Denoisers/Reblur_Specular.hpp"
    "Source/Denoisers/Reblur_SpecularOcclusion.hpp"
    "Source/Denoisers/Reblur_SpecularSh.hpp"
    "Source/Denoisers/Reference.hpp"
    "Source/Denoisers/Relax_Diffuse.hpp"
    "Source/Denoisers/Relax_DiffuseSh.hpp"
    "Source/Denoisers/Relax_DiffuseSpecular.hpp"
    "Source/Denoisers/Relax_DiffuseSpecularSh.hpp"
    "Source/Denoisers/Relax_Specular.hpp"
    "Source/Denoisers/Relax_SpecularSh.hpp"
    "Source/Denoisers/Sigma_Shadow.hpp"
    "Source/Denoisers/Sigma_ShadowTranslucency.hpp"
)
source_group("Denoisers" FILES ${GLOB_DENOISERS})

set(GLOB_RESOURCES
    "Resources/NRD.rc"
    "Resources/Version.h"
)
source_group("Resources" FILES ${GLOB_RESOURCES})

if(NRD_STATIC_LIBRARY)
    add_library(NRD STATIC
        ${GLOB_SOURCE}
        ${GLOB_DENOISERS}
        ${GLOB_RESOURCES}
        ${GLOB_INCUDE}
    )
else()
    add_library(NRD SHARED
        ${GLOB_SOURCE}
        ${GLOB_DENOISERS}
        ${GLOB_RESOURCES}
        ${GLOB_INCUDE}
    )

    if(WIN32)
        target_compile_definitions(NRD PRIVATE "NRD_API=extern \"C\" __declspec(dllexport)")
    else()
        target_compile_definitions(NRD PRIVATE "NRD_API=extern \"C\" __attribute__((visibility(\"default\")))")
    endif()

    if(MSVC)
        target_link_options(NRD PRIVATE
            $<$<CONFIG:Release>:/DEBUG> # generate a PDB for Release build
            $<$<CONFIG:Release>:/OPT:ICF> # perform identical COMDAT folding (removes duplicate code/data)
        )
    endif()
endif()

if("${NRD_SHADERS_PATH}" STREQUAL "")
    set(NRD_SHADERS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/_Shaders")
endif()

target_link_libraries(NRD
    PRIVATE
        MathLib
        ShaderMakeBlob
)
target_include_directories(NRD
    PUBLIC
        "Include"
    PRIVATE
        ${NRD_SHADERS_PATH}
)
target_compile_definitions(NRD
    PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _CRT_SECURE_NO_WARNINGS
        NRD_EMBEDS_SPIRV_SHADERS=$<BOOL:${NRD_EMBEDS_SPIRV_SHADERS}>
        NRD_EMBEDS_DXIL_SHADERS=$<BOOL:${NRD_EMBEDS_DXIL_SHADERS}>
        NRD_EMBEDS_DXBC_SHADERS=$<BOOL:${NRD_EMBEDS_DXBC_SHADERS}>
        SPIRV_SREG_OFFSET=${SPIRV_SREG_OFFSET}
        SPIRV_BREG_OFFSET=${SPIRV_BREG_OFFSET}
        SPIRV_UREG_OFFSET=${SPIRV_UREG_OFFSET}
        SPIRV_TREG_OFFSET=${SPIRV_TREG_OFFSET}
        ${COMPILE_DEFINITIONS}
    PUBLIC
        NRD_STATIC_LIBRARY=$<BOOL:${NRD_STATIC_LIBRARY}>
)
target_compile_options(NRD
    PRIVATE
        # GCC/Clang
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:
            -Wextra
            -Wno-missing-field-initializers
            $<$<NOT:$<BOOL:${IS_ARM64}>>:-mssse3>
        >
        # GCC/Clang, but not over MSVC
        $<$<AND:$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>,$<NOT:$<BOOL:${MSVC}>>>:
            -Werror
            -fPIC
            -fvisibility=hidden
        >
        # MSVC specific options
        $<$<CXX_COMPILER_ID:MSVC>:
            /W4
            /WX
            /wd4324
            $<$<CONFIG:Release>:/Zi>
        >
)
target_compile_features(NRD PUBLIC cxx_std_17)

set_target_properties(NRD PROPERTIES
    FOLDER "NRD"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

# NRD integration
set(GLOB_INTEGRATION
    "Integration/NRDIntegration.h"
    "Integration/NRDIntegration.hpp"
)
source_group("" FILES ${GLOB_INTEGRATION})

add_library(NRDIntegration INTERFACE)
target_sources(NRDIntegration INTERFACE ${GLOB_INTEGRATION})
target_include_directories(NRDIntegration INTERFACE "Integration")
set_target_properties(NRDIntegration PROPERTIES FOLDER "NRD")

# Shaders
file(GLOB_RECURSE SHADERS
    "Shaders/*.hlsl"
    "Shaders/*.hlsli"
)

set_source_files_properties(${SHADERS} PROPERTIES VS_TOOL_OVERRIDE "None")

# ShaderMake general arguments
get_target_property(ML_SOURCE_DIR MathLib SOURCE_DIR)

set(SHADERMAKE_GENERAL_ARGS
    --flatten
    --stripReflection
    --WX
    --sRegShift ${SPIRV_SREG_OFFSET}
    --bRegShift ${SPIRV_BREG_OFFSET}
    --uRegShift ${SPIRV_UREG_OFFSET}
    --tRegShift ${SPIRV_TREG_OFFSET}
    --headerBlob
    --allResourcesBound
    --vulkanVersion 1.2
    --sourceDir "Shaders"
    --ignoreConfigDir
    -c "Shaders/Shaders.cfg"
    -o "${NRD_SHADERS_PATH}"
    -I "${ML_SOURCE_DIR}"
    -I "Shaders/Include"
    -D NRD_INTERNAL
)

if(SHADERMAKE_PATH)
    set(SHADERMAKE_GENERAL_ARGS ${SHADERMAKE_GENERAL_ARGS} --project "NRD" --compactProgress)
else()
    # Needed only to support integrations in products with old ShaderMake
    set(SHADERMAKE_PATH "ShaderMake")
    set(SHADERMAKE_DXC_PATH ${DXC_PATH})
    set(SHADERMAKE_DXC_VK_PATH ${DXC_SPIRV_PATH})
    set(SHADERMAKE_FXC_PATH ${FXC_PATH})
endif()

# ShaderMake commands for each shader code container
set(SHADERMAKE_COMMANDS)
if(NRD_EMBEDS_DXIL_SHADERS)
    list(APPEND SHADERMAKE_COMMANDS COMMAND ${SHADERMAKE_PATH} -p DXIL --compiler "${SHADERMAKE_DXC_PATH}" ${SHADERMAKE_GENERAL_ARGS})
    message(STATUS "NRD_EMBEDS_DXIL_SHADERS")
endif()
if(NRD_EMBEDS_SPIRV_SHADERS)
    list(APPEND SHADERMAKE_COMMANDS COMMAND ${SHADERMAKE_PATH} -p SPIRV --compiler "${SHADERMAKE_DXC_VK_PATH}" ${SHADERMAKE_GENERAL_ARGS})
    message(STATUS "NRD_EMBEDS_SPIRV_SHADERS")
endif()
if(NRD_EMBEDS_DXBC_SHADERS)
    list(APPEND SHADERMAKE_COMMANDS COMMAND ${SHADERMAKE_PATH} -p DXBC --compiler "${SHADERMAKE_FXC_PATH}" ${SHADERMAKE_GENERAL_ARGS})
    message(STATUS "NRD_EMBEDS_DXBC_SHADERS")
endif()
list(APPEND SHADERMAKE_COMMANDS "") # fix for "The system cannot find the batch label specified - VCEnd"

# Add the target with the commands
add_custom_target(NRDShaders ALL ${SHADERMAKE_COMMANDS}
    DEPENDS ShaderMake
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    SOURCES ${SHADERS}
    VERBATIM
)

set_target_properties(NRDShaders PROPERTIES FOLDER "NRD")
add_dependencies(NRD NRDShaders)

# Info
message("NRD: shaders path '${NRD_SHADERS_PATH}'")
message("NRD: output path '${CMAKE_RUNTIME_OUTPUT_DIRECTORY}'")
message("NRD: MathLib path '${ML_SOURCE_DIR}'")
