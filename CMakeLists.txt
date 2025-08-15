cmake_minimum_required(VERSION 3.22...3.30)

include(CMakeDependentOption)
include(FetchContent)

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

# Cached
if(PROJECT_IS_TOP_LEVEL)
    get_property(IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

    if(IS_MULTI_CONFIG)
        set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "")
    endif()

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/_Bin" CACHE STRING "")
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
message(STATUS "NRD_NORMAL_ENCODING = ${NRD_NORMAL_ENCODING}")
message(STATUS "NRD_ROUGHNESS_ENCODING = ${NRD_ROUGHNESS_ENCODING}")

project(NRD VERSION ${NRD_VERSION_MAJOR}.${NRD_VERSION_MINOR}.${NRD_VERSION_BUILD} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable grouping of source files into folders in IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Generate "NRDEncoding.hlsli"
file(WRITE Shaders/Include/NRDEncoding.hlsli
    "// This file is auto-generated during project deployment. Do not modify!\n"
    "#define NRD_NORMAL_ENCODING ${NRD_NORMAL_ENCODING}\n"
    "#define NRD_ROUGHNESS_ENCODING ${NRD_ROUGHNESS_ENCODING}\n")

# Compile options
if(NOT ((CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")))
    set(SIMD -mssse3)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(COMPILE_OPTIONS ${SIMD} -Wextra)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(COMPILE_OPTIONS ${SIMD} -Wextra)
elseif(MSVC)
    set(COMPILE_OPTIONS
        /W4 /WX
        /wd4324 # padding was added at the end of a structure because you specified an alignment specifier
        $<$<CONFIG:Release>:/Zi> # generate a PDB for Release build
    )
else()
    message(WARNING "Unknown compiler!")
endif()

# SPIRV offsets (must respect shader resources limits)
set(SPIRV_SREG_OFFSET 0)
set(SPIRV_BREG_OFFSET 2)
set(SPIRV_UREG_OFFSET 3)
set(SPIRV_TREG_OFFSET 20)

# Compile definitions (public)
set(COMPILE_DEFINITIONS
    NRD_NORMAL_ENCODING=${NRD_NORMAL_ENCODING}
    NRD_ROUGHNESS_ENCODING=${NRD_ROUGHNESS_ENCODING}
)

function(add_compile_definition DEFINITION)
    if(${DEFINITION})
        message(STATUS "${DEFINITION}")
        set(COMPILE_DEFINITIONS ${COMPILE_DEFINITIONS} ${DEFINITION}=1 PARENT_SCOPE)
    else()
        set(COMPILE_DEFINITIONS ${COMPILE_DEFINITIONS} ${DEFINITION}=0 PARENT_SCOPE)
    endif()
endfunction()

add_compile_definition(NRD_STATIC_LIBRARY)
add_compile_definition(NRD_SUPPORTS_VIEWPORT_OFFSET)
add_compile_definition(NRD_SUPPORTS_CHECKERBOARD)
add_compile_definition(NRD_SUPPORTS_HISTORY_CONFIDENCE)
add_compile_definition(NRD_SUPPORTS_DISOCCLUSION_THRESHOLD_MIX)
add_compile_definition(NRD_SUPPORTS_BASECOLOR_METALNESS)
add_compile_definition(NRD_SUPPORTS_ANTIFIREFLY)
add_compile_definition(REBLUR_PERFORMANCE_MODE)

# Download dependencies
set(DEPS)

if(NOT TARGET ShaderMake)
    # ShaderMake
    option(SHADERMAKE_TOOL "" ON)

    FetchContent_Declare(
        shadermake
        GIT_REPOSITORY https://github.com/NVIDIA-RTX/ShaderMake.git
        GIT_TAG 757594f0539314467f74950c8d898e815ecbd383
        GIT_SHALLOW 1
    )
    list(APPEND DEPS shadermake)
endif()

if(NOT TARGET MathLib)
    # MathLib
    FetchContent_Declare(
        mathlib
        GIT_REPOSITORY https://github.com/NVIDIA-RTX/MathLib.git
        GIT_TAG ca2684e5400368628d46bdf67e2c08e33e2c4939
        GIT_SHALLOW 1
    )
    list(APPEND DEPS mathlib)
endif()

if(NOT TARGET NRI AND NRD_NRI)
    # Not needed...
    option(NRI_ENABLE_NVAPI "" OFF)
    option(NRI_ENABLE_AMDAGS "" OFF)
    option(NRI_ENABLE_NVTX_SUPPORT "" OFF)
    option(NRI_ENABLE_NONE_SUPPORT "" OFF)

    # Not needed if no shaders
    if(NOT NRD_EMBEDS_SPIRV_SHADERS)
        option(NRI_ENABLE_VK_SUPPORT "" OFF)
    endif()

    if(NOT NRD_EMBEDS_DXIL_SHADERS)
        option(NRI_ENABLE_D3D12_SUPPORT "" OFF)
    endif()

    if(NOT NRD_EMBEDS_DXBC_SHADERS)
        option(NRI_ENABLE_D3D11_SUPPORT "" OFF)
    endif()

    # NRI
    FetchContent_Declare(
        nri
        GIT_REPOSITORY https://github.com/NVIDIA-RTX/NRI.git
        GIT_TAG 64b4c764aa6b7893df36dbeaccc3aad56828f4f0
        GIT_SHALLOW 1
    )
    list(APPEND DEPS nri)
endif()

if(DEPS)
    message("NRD: downloading dependencies:")
    message(STATUS "${DEPS} ...")

    FetchContent_MakeAvailable(${DEPS})
endif()

# NRD
file(GLOB GLOB_INCUDE "Include/*")
source_group("Include" FILES ${GLOB_INCUDE})

file(GLOB GLOB_SOURCE "Source/*.cpp" "Source/*.h" "Source/*.hpp")
source_group("Source" FILES ${GLOB_SOURCE})

file(GLOB GLOB_DENOISERS "Source/Denoisers/*.cpp" "Source/Denoisers/*.h" "Source/Denoisers/*.hpp")
source_group("Denoisers" FILES ${GLOB_DENOISERS})

file(GLOB GLOB_RESOURCES "Resources/*")
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
else()
    set(NRD_SHADER_BINARIES "--binary")
endif()

target_link_libraries(NRD PRIVATE MathLib)
target_include_directories(NRD PUBLIC "Include")
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
    PUBLIC
        ${COMPILE_DEFINITIONS}
)
target_compile_options(NRD PRIVATE ${COMPILE_OPTIONS})

set_target_properties(NRD PROPERTIES
    FOLDER "NRD"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

# NRD integration
file(GLOB GLOB_INTEGRATION "Integration/*")
source_group("" FILES ${GLOB_INTEGRATION})

if(WIN32)
    add_library(NRDIntegration INTERFACE ${GLOB_INTEGRATION})
    set_target_properties(NRDIntegration PROPERTIES FOLDER "NRD")
else()
    add_library(NRDIntegration INTERFACE) # TODO: fixme...
endif()

target_include_directories(NRDIntegration INTERFACE "Integration")

# Shaders
target_include_directories(NRD PRIVATE "${NRD_SHADERS_PATH}")

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
    --header ${NRD_SHADER_BINARIES}
    --allResourcesBound
    --vulkanVersion 1.2
    --sourceDir "Shaders/Source"
    --ignoreConfigDir
    -c "Shaders/Shaders.cfg"
    -o "${NRD_SHADERS_PATH}"
    -I "${ML_SOURCE_DIR}"
    -I "Shaders/Include"
    -I "Shaders/Resources"
    -D NRD_NORMAL_ENCODING=${NRD_NORMAL_ENCODING}
    -D NRD_ROUGHNESS_ENCODING=${NRD_ROUGHNESS_ENCODING}
    -D NRD_SUPPORTS_VIEWPORT_OFFSET=$<BOOL:${NRD_SUPPORTS_VIEWPORT_OFFSET}>
    -D NRD_SUPPORTS_CHECKERBOARD=$<BOOL:${NRD_SUPPORTS_CHECKERBOARD}>
    -D NRD_SUPPORTS_HISTORY_CONFIDENCE=$<BOOL:${NRD_SUPPORTS_HISTORY_CONFIDENCE}>
    -D NRD_SUPPORTS_DISOCCLUSION_THRESHOLD_MIX=$<BOOL:${NRD_SUPPORTS_DISOCCLUSION_THRESHOLD_MIX}>
    -D NRD_SUPPORTS_BASECOLOR_METALNESS=$<BOOL:${NRD_SUPPORTS_BASECOLOR_METALNESS}>
    -D NRD_SUPPORTS_ANTIFIREFLY=$<BOOL:${NRD_SUPPORTS_ANTIFIREFLY}>
    -D REBLUR_PERFORMANCE_MODE=$<BOOL:${REBLUR_PERFORMANCE_MODE}>
    -D NRD_INTERNAL
)

if(SHADERMAKE_PATH)
    set(SHADERMAKE_GENERAL_ARGS ${SHADERMAKE_GENERAL_ARGS} --project "NRD" --compactProgress)
else()
    # TODO: needed only to support integrations in products with old ShaderMake
    set(SHADERMAKE_PATH "ShaderMake")
    set(SHADERMAKE_DXC_PATH ${DXC_PATH})
    set(SHADERMAKE_DXC_VK_PATH ${DXC_SPIRV_PATH})
    set(SHADERMAKE_FXC_PATH ${FXC_PATH})
endif()

# ShaderMake commands for each shader code container
set(SHADERMAKE_COMMANDS "")

if(NRD_EMBEDS_DXIL_SHADERS)
    set(SHADERMAKE_COMMANDS ${SHADERMAKE_COMMANDS} COMMAND ${SHADERMAKE_PATH} -p DXIL --compiler "${SHADERMAKE_DXC_PATH}" ${SHADERMAKE_GENERAL_ARGS})
    message(STATUS "NRD_EMBEDS_DXIL_SHADERS")
endif()

if(NRD_EMBEDS_SPIRV_SHADERS)
    set(SHADERMAKE_COMMANDS ${SHADERMAKE_COMMANDS} COMMAND ${SHADERMAKE_PATH} -p SPIRV --compiler "${SHADERMAKE_DXC_VK_PATH}" ${SHADERMAKE_GENERAL_ARGS})
    message(STATUS "NRD_EMBEDS_SPIRV_SHADERS")
endif()

if(NRD_EMBEDS_DXBC_SHADERS)
    set(SHADERMAKE_COMMANDS ${SHADERMAKE_COMMANDS} COMMAND ${SHADERMAKE_PATH} -p DXBC --compiler "${SHADERMAKE_FXC_PATH}" ${SHADERMAKE_GENERAL_ARGS})
    message(STATUS "NRD_EMBEDS_DXBC_SHADERS")
endif()

# Add the target with the commands
add_custom_target(NRDShaders ALL ${SHADERMAKE_COMMANDS}
    DEPENDS ShaderMake
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    VERBATIM
    SOURCES ${SHADERS}
)

set_target_properties(NRDShaders PROPERTIES FOLDER "NRD")
add_dependencies(NRD NRDShaders)

# Info
message("NRD: shaders path '${NRD_SHADERS_PATH}'")
message("NRD: output path '${CMAKE_RUNTIME_OUTPUT_DIRECTORY}'")
message("NRD: MathLib path '${ML_SOURCE_DIR}'")
