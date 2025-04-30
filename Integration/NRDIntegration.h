/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

// IMPORTANT: these files must be included beforehand:
//    NRD.h
//    NRI.h
//    Extensions/NRIHelper.h
//    Extensions/NRIWrapperD3D11.h
//    Extensions/NRIWrapperD3D12.h
//    Extensions/NRIWrapperVK.h

#include <array>
#include <vector>
#include <map>
#include <stdio.h>

#define NRD_INTEGRATION_MAJOR 1
#define NRD_INTEGRATION_MINOR 17
#define NRD_INTEGRATION_DATE "30 April 2025"
#define NRD_INTEGRATION 1

// Debugging
#define NRD_INTEGRATION_DEBUG_LOGGING 0

#ifndef NRD_INTEGRATION_ASSERT
    #include <assert.h>
    #define NRD_INTEGRATION_ASSERT(expr, msg) assert(msg && expr)
#endif

#define NRD_INTEGRATION_ABORT_ON_FAILURE(result) if ((result) != nri::Result::SUCCESS) NRD_INTEGRATION_ASSERT(false, "Abort on failure!")

namespace nrd
{

// "TextureBarrierDesc::texture" represents the resource, the rest represents the state ("before" state is unused and can be zeroed)
typedef std::array<nri::TextureBarrierDesc*, (size_t)ResourceType::MAX_NUM - 2> UserPool;

// User pool must contain valid entries for resources, which are required for requested denoisers,
// but the entire pool must be zero-ed during initialization
inline void Integration_SetResource(UserPool& pool, ResourceType slot, nri::TextureBarrierDesc* texture)
{
    NRD_INTEGRATION_ASSERT(texture != nullptr, "Invalid texture!");

    pool[(size_t)slot] = texture;
}

struct IntegrationCreationDesc
{
    // Not so long name
    const char* name = "";

    // Resource dimensions
    uint16_t resourceWidth = 0;
    uint16_t resourceHeight = 0;

    // (1-3) the application must provide number of queued frames, it's needed to guarantee
    // that constant data and descriptor sets are not overwritten while being executed on the GPU
    uint8_t queuedFrameNum = 3;

    // true - enables descriptor caching for the whole lifetime of an Integration instance
    // false - descriptors are cached only within a single "Denoise" call
    bool enableDescriptorCaching = false;

    // Demote FP32 to FP16 (slightly improves performance in exchange of precision loss)
    // (FP32 is used only for viewZ under the hood, all denoisers are FP16 compatible)
    bool demoteFloat32to16 = false;

    // Promote FP16 to FP32 (overkill, kills performance)
    bool promoteFloat16to32 = false;
};

class Integration
{
public:
    inline Integration()
    {}

    inline ~Integration()
    { NRD_INTEGRATION_ASSERT(m_iCore == nullptr, "'m_iCore' must be NULL at this point!"); }

    // There is no "Resize" functionality, because NRD full recreation costs nothing.
    // The main cost comes from render targets resizing, which needs to be done in any case
    // (call Destroy beforehand)
    bool Initialize(const IntegrationCreationDesc& nrdIntegrationDesc, const InstanceCreationDesc& instanceCreationDesc, nri::Device& device, const nri::CoreInterface& iCore);

    // Must be called once on a frame start
    void NewFrame();

    // Explicitly calls eponymous NRD API functions
    bool SetCommonSettings(const CommonSettings& commonSettings);
    bool SetDenoiserSettings(Identifier denoiser, const void* denoiserSettings);

    // Invokes denoising for specified denoisers
    // At the end of the function "nri::TextureBarrierDesc" from "userPool" will represent "final"
    // state of a resource, which must be used as "before" state in the next "barrier" call. The initial
    // state of resources in "userPool" can be restored by using "restoreInitialState = true" (redundant
    // transition barriers will be emitted). This function binds own descriptor heap / pool.
    void Denoise(const Identifier* denoisers, uint32_t denoisersNum, nri::CommandBuffer& commandBuffer, UserPool& userPool, bool restoreInitialState = false);

    // This function assumes that the device is in the IDLE state, i.e. there is no work in flight
    void Destroy();

    // Should not be called explicitly, unless you want to reload pipelines
    void CreatePipelines();

    // Helpers
    inline double GetTotalMemoryUsageInMb() const
    { return double(m_PermanentPoolSize + m_TransientPoolSize) / (1024.0 * 1024.0); }

    inline double GetPersistentMemoryUsageInMb() const
    { return double(m_PermanentPoolSize) / (1024.0 * 1024.0); }

    inline double GetAliasableMemoryUsageInMb() const
    { return double(m_TransientPoolSize) / (1024.0 * 1024.0); }

private:
    Integration(const Integration&) = delete;

    void CreateResources(uint16_t resourceWidth, uint16_t resourceHeight);
    void AllocateAndBindMemory();
    void Dispatch(nri::CommandBuffer& commandBuffer, nri::DescriptorPool& descriptorPool, const DispatchDesc& dispatchDesc, UserPool& userPool);

private:
    std::vector<nri::TextureBarrierDesc> m_TexturePool;
    std::map<uint64_t, nri::Descriptor*> m_CachedDescriptors;
    std::vector<std::vector<nri::Descriptor*>> m_DescriptorsInFlight;
    std::vector<nri::PipelineLayout*> m_PipelineLayouts;
    std::vector<nri::Pipeline*> m_Pipelines;
    std::vector<nri::Memory*> m_MemoryAllocations;
    std::vector<nri::Descriptor*> m_Samplers;
    std::vector<nri::DescriptorPool*> m_DescriptorPools = {};
    std::vector<nri::DescriptorSet*> m_DescriptorSetSamplers = {};
    nri::HelperInterface m_iHelper = {};
    const nri::CoreInterface* m_iCore = nullptr;
    nri::Device* m_Device = nullptr;
    nri::Buffer* m_ConstantBuffer = nullptr;
    nri::Descriptor* m_ConstantBufferView = nullptr;
    FILE* m_Log = nullptr;
    Instance* m_Instance = nullptr;
    uint64_t m_PermanentPoolSize = 0;
    uint64_t m_TransientPoolSize = 0;
    uint64_t m_ConstantBufferSize = 0;
    uint32_t m_ConstantBufferViewSize = 0;
    uint32_t m_ConstantBufferOffset = 0;
    uint32_t m_ConstantBufferOffsetPrev = 0;
    uint32_t m_DescriptorPoolIndex = 0;
    uint32_t m_FrameIndex = uint32_t(-1); // 0 needed after 1st "NewFrame"
    uint32_t m_PrevFrameIndexFromSettings = 0;
    uint16_t m_Width = 0;
    uint16_t m_Height = 0;
    uint8_t m_QueuedFrameNum = 0;
    char m_Name[32] = {};
    bool m_ReloadShaders = false;
    bool m_EnableDescriptorCaching = false;
    bool m_DemoteFloat32to16 = false;
    bool m_PromoteFloat16to32 = false;
};

}
