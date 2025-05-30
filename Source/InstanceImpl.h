/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <cstring> // memset

#include "NRD.h"

typedef nrd::AllocationCallbacks AllocationCallbacks;
#include "StdAllocator.h"

#include "Timer.h"
#include "ml.h"
#include "ml.hlsli"

#define _NRD_STRINGIFY(s) #s
#define NRD_STRINGIFY(s)  _NRD_STRINGIFY(s)

#ifdef NRD_EMBEDS_DXBC_SHADERS
#    define GET_DXBC_SHADER_DESC(shaderName) {g_##shaderName##_cs_dxbc, GetCountOf(g_##shaderName##_cs_dxbc)}
#else
#    define GET_DXBC_SHADER_DESC(shaderName) \
        { \
        }
#endif

#ifdef NRD_EMBEDS_DXIL_SHADERS
#    define GET_DXIL_SHADER_DESC(shaderName) {g_##shaderName##_cs_dxil, GetCountOf(g_##shaderName##_cs_dxil)}
#else
#    define GET_DXIL_SHADER_DESC(shaderName) \
        { \
        }
#endif

#ifdef NRD_EMBEDS_SPIRV_SHADERS
#    define GET_SPIRV_SHADER_DESC(shaderName) {g_##shaderName##_cs_spirv, GetCountOf(g_##shaderName##_cs_spirv)}
#else
#    define GET_SPIRV_SHADER_DESC(shaderName) \
        { \
        }
#endif

#define AddDispatch(shaderName, passName, downsampleFactor) \
    AddComputeDispatchDesc(NumThreads(passName##GroupX, passName##GroupY), \
        downsampleFactor, sizeof(passName##Constants), 1, #shaderName ".cs", \
        GET_DXBC_SHADER_DESC(shaderName), GET_DXIL_SHADER_DESC(shaderName), GET_SPIRV_SHADER_DESC(shaderName))

#define AddDispatchNoConstants(shaderName, passName, downsampleFactor) \
    AddComputeDispatchDesc(NumThreads(passName##GroupX, passName##GroupY), \
        downsampleFactor, 0, 1, #shaderName ".cs", \
        GET_DXBC_SHADER_DESC(shaderName), GET_DXIL_SHADER_DESC(shaderName), GET_SPIRV_SHADER_DESC(shaderName))

#define AddDispatchRepeated(shaderName, passName, downsampleFactor, repeatNum) \
    AddComputeDispatchDesc(NumThreads(passName##GroupX, passName##GroupY), \
        downsampleFactor, sizeof(passName##Constants), repeatNum, #shaderName ".cs", \
        GET_DXBC_SHADER_DESC(shaderName), GET_DXIL_SHADER_DESC(shaderName), GET_SPIRV_SHADER_DESC(shaderName))

#define PushPass(passName) \
    _PushPass(NRD_STRINGIFY(DENOISER_NAME) " - " passName)

// TODO: rework is needed, but still better than copy-pasting
#define NRD_DECLARE_DIMS \
    [[maybe_unused]] uint16_t resourceW = m_CommonSettings.resourceSize[0]; \
    [[maybe_unused]] uint16_t resourceH = m_CommonSettings.resourceSize[1]; \
    [[maybe_unused]] uint16_t resourceWprev = m_CommonSettings.resourceSizePrev[0]; \
    [[maybe_unused]] uint16_t resourceHprev = m_CommonSettings.resourceSizePrev[1]; \
    [[maybe_unused]] uint16_t rectW = m_CommonSettings.rectSize[0]; \
    [[maybe_unused]] uint16_t rectH = m_CommonSettings.rectSize[1]; \
    [[maybe_unused]] uint16_t rectWprev = m_CommonSettings.rectSizePrev[0]; \
    [[maybe_unused]] uint16_t rectHprev = m_CommonSettings.rectSizePrev[1];

// IMPORTANT: needed only for DXBC
#undef BYTE
#define BYTE uint8_t

// Macro magic for shared headers
// IMPORTANT: do not use "float3" constants because of sizeof( ml::float3 ) = 16!
#define NRD_CONSTANTS_START(name) struct name {
#define NRD_CONSTANT(type, name)  type name;
#define NRD_CONSTANTS_END \
    } \
    ;

#define NRD_INPUTS_START
#define NRD_INPUT(...)
#define NRD_INPUTS_END
#define NRD_OUTPUTS_START
#define NRD_OUTPUT(...)
#define NRD_OUTPUTS_END
#define NRD_SAMPLERS_START
#define NRD_SAMPLER(...)
#define NRD_SAMPLERS_END

typedef uint32_t uint;

// Implementation
namespace nrd {
constexpr uint16_t PERMANENT_POOL_START = 1000;
constexpr uint16_t TRANSIENT_POOL_START = 2000;
constexpr size_t CONSTANT_DATA_SIZE = 128 * 1024; // TODO: improve

constexpr uint16_t USE_MAX_DIMS = 0xFFFF;
constexpr uint16_t IGNORE_RS = 0xFFFE;

inline uint16_t DivideUp(uint32_t x, uint16_t y) {
    return uint16_t((x + y - 1) / y);
}

template <class T>
inline uint16_t AsUint(T x) {
    return (uint16_t)x;
}

union Settings {
    ReblurSettings reblur;
    RelaxSettings relax;
    SigmaSettings sigma;
    ReferenceSettings reference;
};

struct DenoiserData {
    DenoiserDesc desc;
    Settings settings;
    size_t settingsSize;
    size_t dispatchOffset;
    size_t pingPongOffset;
    size_t pingPongNum;
};

struct PingPong {
    size_t resourceIndex;
    uint16_t indexInPoolToSwapWith;
};

struct NumThreads {
    inline NumThreads(uint8_t w, uint8_t h)
        : width(w), height(h) {
    }

    inline NumThreads()
        : width(0), height(0) {
    }

    uint8_t width;
    uint8_t height;
};

struct InternalDispatchDesc {
    const char* name;
    const ResourceDesc* resources; // concatenated resources for all "ResourceRangeDesc" descriptions in InstanceDesc::pipelines[ pipelineIndex ]
    uint32_t resourcesNum;
    const uint8_t* constantBufferData;
    uint32_t constantBufferDataSize;
    Identifier identifier;
    uint16_t pipelineIndex;
    uint16_t downsampleFactor;
    uint16_t maxRepeatNum; // IMPORTANT: must be same for all permutations (i.e. for same "name")
    NumThreads numThreads;
};

struct ClearResource {
    Identifier identifier;
    ResourceDesc resource;
    uint16_t downsampleFactor;
    bool isInteger;
};

class InstanceImpl {
    // Add denoisers here
public:
    // Reblur
    void Add_ReblurDiffuse(DenoiserData& denoiserData);
    void Add_ReblurDiffuseOcclusion(DenoiserData& denoiserData);
    void Add_ReblurDiffuseSh(DenoiserData& denoiserData);
    void Add_ReblurSpecular(DenoiserData& denoiserData);
    void Add_ReblurSpecularOcclusion(DenoiserData& denoiserData);
    void Add_ReblurSpecularSh(DenoiserData& denoiserData);
    void Add_ReblurDiffuseSpecular(DenoiserData& denoiserData);
    void Add_ReblurDiffuseSpecularOcclusion(DenoiserData& denoiserData);
    void Add_ReblurDiffuseSpecularSh(DenoiserData& denoiserData);
    void Add_ReblurDiffuseDirectionalOcclusion(DenoiserData& denoiserData);
    void Update_Reblur(const DenoiserData& denoiserData);
    void Update_ReblurOcclusion(const DenoiserData& denoiserData);
    void AddSharedConstants_Reblur(const ReblurSettings& settings, void* data);

    // Relax
    void Add_RelaxDiffuse(DenoiserData& denoiserData);
    void Add_RelaxDiffuseSh(DenoiserData& denoiserData);
    void Add_RelaxSpecular(DenoiserData& denoiserData);
    void Add_RelaxSpecularSh(DenoiserData& denoiserData);
    void Add_RelaxDiffuseSpecular(DenoiserData& denoiserData);
    void Add_RelaxDiffuseSpecularSh(DenoiserData& denoiserData);
    void Update_Relax(const DenoiserData& denoiserData);
    void AddSharedConstants_Relax(const RelaxSettings& settings, void* data);

    // Sigma
    void Add_SigmaShadow(DenoiserData& denoiserData);
    void Add_SigmaShadowTranslucency(DenoiserData& denoiserData);
    void Update_SigmaShadow(const DenoiserData& denoiserData);
    void AddSharedConstants_Sigma(const SigmaSettings& settings, void* data);

    // Other
    void Add_Reference(DenoiserData& denoiserData);
    void Update_Reference(const DenoiserData& denoiserData);

    // Internal
public:
    inline InstanceImpl(const StdAllocator<uint8_t>& stdAllocator)
        : m_StdAllocator(stdAllocator)
        , m_DenoiserData(GetStdAllocator())
        , m_PermanentPool(GetStdAllocator())
        , m_TransientPool(GetStdAllocator())
        , m_Resources(GetStdAllocator())
        , m_ClearResources(GetStdAllocator())
        , m_PingPongs(GetStdAllocator())
        , m_ResourceRanges(GetStdAllocator())
        , m_Pipelines(GetStdAllocator())
        , m_Dispatches(GetStdAllocator())
        , m_ActiveDispatches(GetStdAllocator())
        , m_IndexRemap(GetStdAllocator()) {
        m_ConstantDataUnaligned = m_StdAllocator.allocate(CONSTANT_DATA_SIZE + sizeof(float4));

        // IMPORTANT: underlying memory for constants must be aligned, as well as any individual SSE-type containing member,
        // because a compiler can generate dangerous "movaps" instruction!
        m_ConstantData = Align(m_ConstantDataUnaligned, sizeof(float4));
        memset(m_ConstantData, 0, CONSTANT_DATA_SIZE);

        m_DenoiserData.reserve(8);
        m_PermanentPool.reserve(32);
        m_TransientPool.reserve(32);
        m_Resources.reserve(128);
        m_ClearResources.reserve(32);
        m_PingPongs.reserve(32);
        m_ResourceRanges.reserve(64);
        m_Pipelines.reserve(32);
        m_Dispatches.reserve(32);
        m_ActiveDispatches.reserve(32);
    }

    ~InstanceImpl() {
        m_StdAllocator.deallocate(m_ConstantDataUnaligned, 0);
    }

    inline const InstanceDesc& GetDesc() const {
        return m_Desc;
    }

    inline StdAllocator<uint8_t>& GetStdAllocator() {
        return m_StdAllocator;
    }

    Result Create(const InstanceCreationDesc& instanceCreationDesc);
    Result SetCommonSettings(const CommonSettings& commonSettings);
    Result SetDenoiserSettings(Identifier identifier, const void* denoiserSettings);
    Result GetComputeDispatches(const Identifier* identifiers, uint32_t identifiersNum, const DispatchDesc*& dispatchDescs, uint32_t& dispatchDescsNum);

private:
    void AddComputeDispatchDesc(
        NumThreads numThreads,
        uint16_t downsampleFactor,
        uint32_t constantBufferDataSize,
        uint32_t maxRepeatNum,
        const char* shaderFileName,
        const ComputeShaderDesc& dxbc,
        const ComputeShaderDesc& dxil,
        const ComputeShaderDesc& spirv);

    void PrepareDesc();
    void UpdatePingPong(const DenoiserData& denoiserData);
    void PushTexture(DescriptorType descriptorType, uint16_t localIndex, uint16_t indexToSwapWith = uint16_t(-1));

    // Available in denoiser implementations
private:
    void AddTextureToTransientPool(const TextureDesc& textureDesc);
    void* PushDispatch(const DenoiserData& denoiserData, uint32_t localIndex);

    inline void AddTextureToPermanentPool(const TextureDesc& textureDesc) {
        m_PermanentPool.push_back(textureDesc);
    }

    inline void PushInput(uint16_t indexInPool, uint16_t indexToSwapWith = uint16_t(-1)) {
        PushTexture(DescriptorType::TEXTURE, indexInPool, indexToSwapWith);
    }

    inline void PushOutput(uint16_t indexInPool, uint16_t indexToSwapWith = uint16_t(-1)) {
        PushTexture(DescriptorType::STORAGE_TEXTURE, indexInPool, indexToSwapWith);
    }

    inline void _PushPass(const char* name) {
        m_PassName = name;
        m_ResourceOffset = m_Resources.size();
    }

private:
    StdAllocator<uint8_t> m_StdAllocator;
    Vector<DenoiserData> m_DenoiserData;
    Vector<TextureDesc> m_PermanentPool;
    Vector<TextureDesc> m_TransientPool;
    Vector<ResourceDesc> m_Resources;
    Vector<ClearResource> m_ClearResources;
    Vector<PingPong> m_PingPongs;
    Vector<ResourceRangeDesc> m_ResourceRanges;
    Vector<PipelineDesc> m_Pipelines;
    Vector<InternalDispatchDesc> m_Dispatches;
    Vector<DispatchDesc> m_ActiveDispatches;
    Vector<uint16_t> m_IndexRemap;
    Timer m_Timer;
    InstanceDesc m_Desc = {};
    CommonSettings m_CommonSettings = {};
    float4x4 m_ViewToClip = float4x4::Identity();
    float4x4 m_ViewToClipPrev = float4x4::Identity();
    float4x4 m_ClipToView = float4x4::Identity();
    float4x4 m_ClipToViewPrev = float4x4::Identity();
    float4x4 m_WorldToView = float4x4::Identity();
    float4x4 m_WorldToViewPrev = float4x4::Identity();
    float4x4 m_ViewToWorld = float4x4::Identity();
    float4x4 m_ViewToWorldPrev = float4x4::Identity();
    float4x4 m_WorldToClip = float4x4::Identity();
    float4x4 m_WorldToClipPrev = float4x4::Identity();
    float4x4 m_ClipToWorld = float4x4::Identity();
    float4x4 m_ClipToWorldPrev = float4x4::Identity();
    float4x4 m_WorldPrevToWorld = float4x4::Identity();
    float4 m_RotatorPre = float4::Zero();
    float4 m_Rotator = float4::Zero();
    float4 m_RotatorPost = float4::Zero();
    float4 m_Frustum = float4::Zero();
    float4 m_FrustumPrev = float4::Zero();
    float3 m_CameraDelta = float3::Zero();
    float3 m_ViewDirection = float3::Zero();
    float3 m_ViewDirectionPrev = float3::Zero();
    float m_SplitScreenPrev = 0.0f;
    const char* m_PassName = nullptr;
    uint8_t* m_ConstantDataUnaligned = nullptr;
    uint8_t* m_ConstantData = nullptr;
    size_t m_ConstantDataOffset = 0;
    size_t m_ResourceOffset = 0;
    size_t m_DispatchClearIndex[2] = {};
    float m_OrthoMode = 0.0f;
    float m_CheckerboardResolveAccumSpeed = 0.0f;
    float m_JitterDelta = 0.0f;
    float m_TimeDelta = 0.0f;
    float m_FrameRateScale = 0.0f;
    float m_ProjectY = 0.0f;
    uint32_t m_AccumulatedFrameNum = 0;
    uint16_t m_TransientPoolOffset = 0;
    uint16_t m_PermanentPoolOffset = 0;
    bool m_IsFirstUse = true;
};
} // namespace nrd
