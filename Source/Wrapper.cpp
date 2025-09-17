/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "../Resources/Version.h"
#include "InstanceImpl.h"
#include "NRD.h"

#include <array>

static_assert(VERSION_MAJOR == NRD_VERSION_MAJOR, "VERSION_MAJOR & NRD_VERSION_MAJOR don't match!");
static_assert(VERSION_MINOR == NRD_VERSION_MINOR, "VERSION_MINOR & NRD_VERSION_MINOR don't match!");
static_assert(VERSION_BUILD == NRD_VERSION_BUILD, "VERSION_BUILD & NRD_VERSION_BUILD don't match!");
static_assert(NRD_NORMAL_ENCODING >= 0 && NRD_NORMAL_ENCODING < (uint32_t)nrd::NormalEncoding::MAX_NUM, "NRD_NORMAL_ENCODING out of bounds!");
static_assert(NRD_ROUGHNESS_ENCODING >= 0 && NRD_ROUGHNESS_ENCODING < (uint32_t)nrd::RoughnessEncoding::MAX_NUM, "NRD_ROUGHNESS_ENCODING out of bounds!");

constexpr std::array<nrd::Denoiser, (size_t)nrd::Denoiser::MAX_NUM> g_NrdSupportedDenoisers = {
    nrd::Denoiser::REBLUR_DIFFUSE,
    nrd::Denoiser::REBLUR_DIFFUSE_OCCLUSION,
    nrd::Denoiser::REBLUR_DIFFUSE_SH,
    nrd::Denoiser::REBLUR_SPECULAR,
    nrd::Denoiser::REBLUR_SPECULAR_OCCLUSION,
    nrd::Denoiser::REBLUR_SPECULAR_SH,
    nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR,
    nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR_OCCLUSION,
    nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR_SH,
    nrd::Denoiser::REBLUR_DIFFUSE_DIRECTIONAL_OCCLUSION,
    nrd::Denoiser::RELAX_DIFFUSE,
    nrd::Denoiser::RELAX_DIFFUSE_SH,
    nrd::Denoiser::RELAX_SPECULAR,
    nrd::Denoiser::RELAX_SPECULAR_SH,
    nrd::Denoiser::RELAX_DIFFUSE_SPECULAR,
    nrd::Denoiser::RELAX_DIFFUSE_SPECULAR_SH,
    nrd::Denoiser::SIGMA_SHADOW,
    nrd::Denoiser::SIGMA_SHADOW_TRANSLUCENCY,
    nrd::Denoiser::REFERENCE,
};

constexpr nrd::LibraryDesc g_NrdLibraryDesc = {
    {SPIRV_SREG_OFFSET, SPIRV_TREG_OFFSET, SPIRV_BREG_OFFSET, SPIRV_UREG_OFFSET},
    g_NrdSupportedDenoisers.data(),
    (uint32_t)g_NrdSupportedDenoisers.size(),
    VERSION_MAJOR,
    VERSION_MINOR,
    VERSION_BUILD,
    (nrd::NormalEncoding)NRD_NORMAL_ENCODING,
    (nrd::RoughnessEncoding)NRD_ROUGHNESS_ENCODING};

const char* g_NrdResourceTypeNames[] = {
    "IN_MV",
    "IN_NORMAL_ROUGHNESS",
    "IN_VIEWZ",
    "IN_DIFF_RADIANCE_HITDIST",
    "IN_SPEC_RADIANCE_HITDIST",
    "IN_DIFF_HITDIST",
    "IN_SPEC_HITDIST",
    "IN_DIFF_DIRECTION_HITDIST",
    "IN_DIFF_SH0",
    "IN_DIFF_SH1",
    "IN_SPEC_SH0",
    "IN_SPEC_SH1",
    "IN_DIFF_CONFIDENCE",
    "IN_SPEC_CONFIDENCE",
    "IN_DISOCCLUSION_THRESHOLD_MIX",
    "IN_BASECOLOR_METALNESS",
    "IN_PENUMBRA",
    "IN_TRANSLUCENCY",
    "IN_SIGNAL",

    "OUT_DIFF_RADIANCE_HITDIST",
    "OUT_SPEC_RADIANCE_HITDIST",
    "OUT_DIFF_SH0",
    "OUT_DIFF_SH1",
    "OUT_SPEC_SH0",
    "OUT_SPEC_SH1",
    "OUT_DIFF_HITDIST",
    "OUT_SPEC_HITDIST",
    "OUT_DIFF_DIRECTION_HITDIST",
    "OUT_SHADOW_TRANSLUCENCY",
    "OUT_SIGNAL",
    "OUT_VALIDATION",

    "TRANSIENT_POOL",
    "PERMANENT_POOL",
};
static_assert(GetCountOf(g_NrdResourceTypeNames) == (uint32_t)nrd::ResourceType::MAX_NUM);

const char* g_NrdDenoiserNames[] = {
    "REBLUR_DIFFUSE",
    "REBLUR_DIFFUSE_OCCLUSION",
    "REBLUR_DIFFUSE_SH",
    "REBLUR_SPECULAR",
    "REBLUR_SPECULAR_OCCLUSION",
    "REBLUR_SPECULAR_SH",
    "REBLUR_DIFFUSE_SPECULAR",
    "REBLUR_DIFFUSE_SPECULAR_OCCLUSION",
    "REBLUR_DIFFUSE_SPECULAR_SH",
    "REBLUR_DIFFUSE_DIRECTIONAL_OCCLUSION",

    "RELAX_DIFFUSE",
    "RELAX_DIFFUSE_SH",
    "RELAX_SPECULAR",
    "RELAX_SPECULAR_SH",
    "RELAX_DIFFUSE_SPECULAR",
    "RELAX_DIFFUSE_SPECULAR_SH",

    "SIGMA_SHADOW",
    "SIGMA_SHADOW_TRANSLUCENCY",

    "REFERENCE",
};
static_assert(GetCountOf(g_NrdDenoiserNames) == (uint32_t)nrd::Denoiser::MAX_NUM);

NRD_API const nrd::LibraryDesc* NRD_CALL nrd::GetLibraryDesc() {
    return &g_NrdLibraryDesc;
}

NRD_API nrd::Result NRD_CALL nrd::CreateInstance(const InstanceCreationDesc& instanceCreationDesc, Instance*& instance) {
    InstanceCreationDesc modifiedInstanceCreationDesc = instanceCreationDesc;
    CheckAndSetDefaultAllocator(modifiedInstanceCreationDesc.allocationCallbacks);

    StdAllocator<uint8_t> memoryAllocator(modifiedInstanceCreationDesc.allocationCallbacks);

    InstanceImpl* implementation = Allocate<InstanceImpl>(memoryAllocator, memoryAllocator);
    Result result = implementation->Create(modifiedInstanceCreationDesc);

    if (result == Result::SUCCESS) {
        instance = (Instance*)implementation;
        return Result::SUCCESS;
    }

    Deallocate(memoryAllocator, implementation);

    return result;
}

NRD_API const nrd::InstanceDesc* NRD_CALL nrd::GetInstanceDesc(const Instance& denoiser) {
    return &((const InstanceImpl&)denoiser).GetDesc();
}

NRD_API nrd::Result NRD_CALL nrd::SetCommonSettings(Instance& instance, const CommonSettings& commonSettings) {
    return ((InstanceImpl&)instance).SetCommonSettings(commonSettings);
}

NRD_API nrd::Result NRD_CALL nrd::SetDenoiserSettings(Instance& instance, Identifier identifier, const void* denoiserSettings) {
    return ((InstanceImpl&)instance).SetDenoiserSettings(identifier, denoiserSettings);
}

NRD_API nrd::Result NRD_CALL nrd::GetComputeDispatches(Instance& instance, const Identifier* identifiers, uint32_t identifiersNum, const DispatchDesc*& dispatchDescs, uint32_t& dispatchDescsNum) {
    return ((InstanceImpl&)instance).GetComputeDispatches(identifiers, identifiersNum, dispatchDescs, dispatchDescsNum);
}

NRD_API void NRD_CALL nrd::DestroyInstance(Instance& instance) {
    StdAllocator<uint8_t> memoryAllocator = ((InstanceImpl&)instance).GetStdAllocator();
    Deallocate(memoryAllocator, (InstanceImpl*)&instance);
}

NRD_API const char* NRD_CALL nrd::GetResourceTypeString(ResourceType resourceType) {
    uint32_t i = (uint32_t)resourceType;

    return i < (uint32_t)ResourceType::MAX_NUM ? g_NrdResourceTypeNames[i] : nullptr;
}

NRD_API const char* NRD_CALL nrd::GetDenoiserString(Denoiser denoiser) {
    uint32_t i = (uint32_t)denoiser;

    return i < (uint32_t)Denoiser::MAX_NUM ? g_NrdDenoiserNames[i] : nullptr;
}
