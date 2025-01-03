/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "../Shaders/Include/NRD.hlsli"
#include "InstanceImpl.h"

#include <assert.h> // assert
#include <array>

constexpr std::array<nrd::Sampler, (size_t)nrd::Sampler::MAX_NUM> g_Samplers =
{
    nrd::Sampler::NEAREST_CLAMP,
    nrd::Sampler::LINEAR_CLAMP,
};

constexpr std::array<bool, (size_t)nrd::Format::MAX_NUM> g_IsIntegerFormat =
{
    false,        // R8_UNORM
    false,        // R8_SNORM
    true,         // R8_UINT
    false,        // R8_SINT
    false,        // RG8_UNORM
    false,        // RG8_SNORM
    true,         // RG8_UINT
    false,        // RG8_SINT
    false,        // RGBA8_UNORM
    false,        // RGBA8_SNORM
    true,         // RGBA8_UINT
    false,        // RGBA8_SINT
    false,        // RGBA8_SRGB
    false,        // R16_UNORM
    false,        // R16_SNORM
    true,         // R16_UINT
    false,        // R16_SINT
    false,        // R16_SFLOAT
    false,        // RG16_UNORM
    false,        // RG16_SNORM
    true,         // RG16_UINT
    false,        // RG16_SINT
    false,        // RG16_SFLOAT
    false,        // RGBA16_UNORM
    false,        // RGBA16_SNORM
    true,         // RGBA16_UINT
    false,        // RGBA16_SINT
    false,        // RGBA16_SFLOAT
    true,         // R32_UINT
    false,        // R32_SINT
    false,        // R32_SFLOAT
    true,         // RG32_UINT
    false,        // RG32_SINT
    false,        // RG32_SFLOAT
    true,         // RGB32_UINT
    false,        // RGB32_SINT
    false,        // RGB32_SFLOAT
    true,         // RGBA32_UINT
    false,        // RGBA32_SINT
    false,        // RGBA32_SFLOAT
    false,        // R10_G10_B10_A2_UNORM
    true,         // R10_G10_B10_A2_UINT
    false,        // R11_G11_B10_UFLOAT
    false,        // R9_G9_B9_E5_UFLOAT
};

#include "../Shaders/Resources/Clear_Float.resources.hlsli"
#include "../Shaders/Resources/Clear_Uint.resources.hlsli"

#ifdef NRD_EMBEDS_DXBC_SHADERS
    #include "Clear_Float.cs.dxbc.h"
    #include "Clear_Uint.cs.dxbc.h"
#endif

#ifdef NRD_EMBEDS_DXIL_SHADERS
    #include "Clear_Float.cs.dxil.h"
    #include "Clear_Uint.cs.dxil.h"
#endif

#ifdef NRD_EMBEDS_SPIRV_SHADERS
    #include "Clear_Float.cs.spirv.h"
    #include "Clear_Uint.cs.spirv.h"
#endif

inline bool IsInList(nrd::Identifier identifier, const nrd::Identifier* identifiers, uint32_t identifiersNum)
{
    for (uint32_t i = 0; i < identifiersNum; i++)
    {
        if (identifiers[i] == identifier)
            return true;
    }

    return false;
}

nrd::Result nrd::InstanceImpl::Create(const InstanceCreationDesc& instanceCreationDesc)
{
    const LibraryDesc& libraryDesc = GetLibraryDesc();

    // Collect dispatches from all denoisers
    for (uint32_t i = 0; i < instanceCreationDesc.denoisersNum; i++)
    {
        const DenoiserDesc& denoiserDesc = instanceCreationDesc.denoisers[i];

        // Check that denoiser is supported
        uint32_t j = 0;
        for (; j < libraryDesc.supportedDenoisersNum; j++)
        {
            if (denoiserDesc.denoiser == libraryDesc.supportedDenoisers[j])
                break;
        }
        if (j == libraryDesc.supportedDenoisersNum)
            return Result::UNSUPPORTED;

        // Check that identifier is unique
        for (j = 0; j < instanceCreationDesc.denoisersNum; j++)
        {
            if (i != j && instanceCreationDesc.denoisers[j].identifier == denoiserDesc.identifier)
                return Result::NON_UNIQUE_IDENTIFIER;
        }

        // Append dispatches for the current denoiser
        m_PermanentPoolOffset = (uint16_t)m_PermanentPool.size();
        m_TransientPoolOffset = (uint16_t)m_TransientPool.size();

        m_IndexRemap.clear();

        DenoiserData denoiserData = {};
        denoiserData.desc = denoiserDesc;
        denoiserData.dispatchOffset = m_Dispatches.size();
        denoiserData.pingPongOffset = m_PingPongs.size();

        size_t resourceOffset = m_Resources.size();

        if (denoiserDesc.denoiser == Denoiser::REBLUR_DIFFUSE)
            Add_ReblurDiffuse(denoiserData);
        else if (denoiserDesc.denoiser == Denoiser::REBLUR_DIFFUSE_OCCLUSION)
            Add_ReblurDiffuseOcclusion(denoiserData);
        else if (denoiserDesc.denoiser == Denoiser::REBLUR_DIFFUSE_SH)
            Add_ReblurDiffuseSh(denoiserData);
        else if (denoiserDesc.denoiser == Denoiser::REBLUR_SPECULAR)
            Add_ReblurSpecular(denoiserData);
        else if (denoiserDesc.denoiser == Denoiser::REBLUR_SPECULAR_OCCLUSION)
            Add_ReblurSpecularOcclusion(denoiserData);
        else if (denoiserDesc.denoiser == Denoiser::REBLUR_SPECULAR_SH)
            Add_ReblurSpecularSh(denoiserData);
        else if (denoiserDesc.denoiser == Denoiser::REBLUR_DIFFUSE_SPECULAR)
            Add_ReblurDiffuseSpecular(denoiserData);
        else if (denoiserDesc.denoiser == Denoiser::REBLUR_DIFFUSE_SPECULAR_OCCLUSION)
            Add_ReblurDiffuseSpecularOcclusion(denoiserData);
        else if (denoiserDesc.denoiser == Denoiser::REBLUR_DIFFUSE_SPECULAR_SH)
            Add_ReblurDiffuseSpecularSh(denoiserData);
        else if (denoiserDesc.denoiser == Denoiser::REBLUR_DIFFUSE_DIRECTIONAL_OCCLUSION)
            Add_ReblurDiffuseDirectionalOcclusion(denoiserData);
        else if (denoiserDesc.denoiser == Denoiser::RELAX_DIFFUSE)
            Add_RelaxDiffuse(denoiserData);
        else if (denoiserDesc.denoiser == Denoiser::RELAX_DIFFUSE_SH)
            Add_RelaxDiffuseSh(denoiserData);
        else if (denoiserDesc.denoiser == Denoiser::RELAX_SPECULAR)
            Add_RelaxSpecular(denoiserData);
        else if (denoiserDesc.denoiser == Denoiser::RELAX_SPECULAR_SH)
            Add_RelaxSpecularSh(denoiserData);
        else if (denoiserDesc.denoiser == Denoiser::RELAX_DIFFUSE_SPECULAR)
            Add_RelaxDiffuseSpecular(denoiserData);
        else if (denoiserDesc.denoiser == Denoiser::RELAX_DIFFUSE_SPECULAR_SH)
            Add_RelaxDiffuseSpecularSh(denoiserData);
        else if (denoiserDesc.denoiser == Denoiser::SIGMA_SHADOW)
            Add_SigmaShadow(denoiserData);
        else if (denoiserDesc.denoiser == Denoiser::SIGMA_SHADOW_TRANSLUCENCY)
            Add_SigmaShadowTranslucency(denoiserData);
        else if (denoiserDesc.denoiser == Denoiser::REFERENCE)
            Add_Reference(denoiserData);
        else // Should not be here
            return Result::INVALID_ARGUMENT;

        denoiserData.pingPongNum = m_PingPongs.size() - denoiserData.pingPongOffset;

        // Patch identifiers
        for (size_t dispatchIndex = denoiserData.dispatchOffset; dispatchIndex < m_Dispatches.size(); dispatchIndex++)
        {
            InternalDispatchDesc& internalDispatchDesc = m_Dispatches[dispatchIndex];
            internalDispatchDesc.identifier = denoiserDesc.identifier;
        }

        // Gather resources, which need to be cleared
        for (size_t resourceIndex = resourceOffset; resourceIndex < m_Resources.size(); resourceIndex++)
        {
            const ResourceDesc& resource = m_Resources[resourceIndex];

            // Loop through all resources and find all used as STORAGE (i.e. ignore read-only user provided inputs)
            if (resource.descriptorType != DescriptorType::STORAGE_TEXTURE)
                continue;

            // Skip "OUT_VALIDATION" resource because it can be not provided
            if (resource.type == ResourceType::OUT_VALIDATION)
                continue;

            // Keep only unique instances
            bool isFound = false;
            for (const ClearResource& temp : m_ClearResources)
            {
                if (temp.resource.descriptorType == resource.descriptorType &&
                    temp.resource.type == resource.type &&
                    temp.resource.indexInPool == resource.indexInPool)
                {
                    isFound = true;
                    break;
                }
            }

            if (!isFound)
            {
                // Is integer?
                bool isInteger = false;
                uint16_t downsampleFactor = 1;
                if (resource.type == ResourceType::PERMANENT_POOL || resource.type == ResourceType::TRANSIENT_POOL)
                {
                    TextureDesc& textureDesc = resource.type == ResourceType::PERMANENT_POOL ? m_PermanentPool[resource.indexInPool] : m_TransientPool[resource.indexInPool];
                    isInteger = g_IsIntegerFormat[(size_t)textureDesc.format];
                    downsampleFactor = textureDesc.downsampleFactor;
                }

                // Add PING resource
                m_ClearResources.push_back( {denoiserDesc.identifier, resource, downsampleFactor, isInteger} );

                // Add PONG resource
                for (uint32_t p = 0; p < denoiserData.pingPongNum; p++)
                {
                    const PingPong& pingPong = m_PingPongs[denoiserData.pingPongOffset + p];
                    if (pingPong.resourceIndex == (uint32_t)resourceIndex)
                    {
                        ResourceDesc resourcePong = {resource.descriptorType, resource.type, pingPong.indexInPoolToSwapWith};
                        m_ClearResources.push_back( {denoiserDesc.identifier, resourcePong, downsampleFactor, isInteger} );
                        break;
                    }
                }
            }
        }

        m_DenoiserData.push_back(denoiserData);
    }

    // Add "clear" dispatches
    m_DispatchClearIndex[0] = m_Dispatches.size();
    _PushPass("Clear (f)");
    {
        PushOutput(0);
        AddDispatchNoConstants( Clear_Float, Clear_Float, 1 );
    }

    m_DispatchClearIndex[1] = m_Dispatches.size();
    _PushPass("Clear (ui)");
    {
        PushOutput(0);
        AddDispatchNoConstants( Clear_Uint, Clear_Uint, 1 );
    }

    PrepareDesc();

    // IMPORTANT: since now all std::vectors become "locked" (no reallocations)

    return Result::SUCCESS;
}

nrd::Result nrd::InstanceImpl::SetCommonSettings(const CommonSettings& commonSettings)
{
    // TODO: matrix verifications?
    bool isValid = commonSettings.viewZScale > 0.0f;
    assert("'viewZScale' can't be <= 0" && isValid);

    isValid &= commonSettings.resourceSize[0] != 0 && commonSettings.resourceSize[1] != 0;
    assert("'resourceSize' can't be 0" && isValid);

    isValid &= commonSettings.resourceSizePrev[0] != 0 && commonSettings.resourceSizePrev[1] != 0;
    assert("'resourceSizePrev' can't be 0" && isValid);

    isValid &= commonSettings.rectSize[0] != 0 && commonSettings.rectSize[1] != 0;
    assert("'rectSize' can't be 0" && isValid);

    isValid &= commonSettings.rectSizePrev[0] != 0 && commonSettings.rectSizePrev[1] != 0;
    assert("'rectSizePrev' can't be 0" && isValid);

    isValid &= ((commonSettings.motionVectorScale[0] != 0.0f && commonSettings.motionVectorScale[1] != 0.0f) || commonSettings.isMotionVectorInWorldSpace);
    assert("'mvScale.xy' can't be 0" && isValid);

    isValid &= commonSettings.cameraJitter[0] >= -0.5f && commonSettings.cameraJitter[0] <= 0.5f && commonSettings.cameraJitter[1] >= -0.5f && commonSettings.cameraJitter[1] <= 0.5f;
    assert("'cameraJitter' must be in range [-0.5; 0.5]" && isValid);

    isValid &= commonSettings.cameraJitterPrev[0] >= -0.5f && commonSettings.cameraJitterPrev[0] <= 0.5f && commonSettings.cameraJitterPrev[1] >= -0.5f && commonSettings.cameraJitterPrev[1] <= 0.5f;
    assert("'cameraJitterPrev' must be in range [-0.5; 0.5]" && isValid);

    isValid &= commonSettings.denoisingRange > 0.0f;
    assert("'denoisingRange' must be >= 0" && isValid);

    isValid &= commonSettings.disocclusionThreshold > 0.0f;
    assert("'disocclusionThreshold' must be > 0" && isValid);

    isValid &= commonSettings.disocclusionThresholdAlternate > 0.0f;
    assert("'disocclusionThresholdAlternate' must be > 0" && isValid);

    isValid &= commonSettings.strandMaterialID != 0.0f || GetLibraryDesc().normalEncoding == NormalEncoding::R10_G10_B10_A2_UNORM;
    assert("'strandMaterialID' can't be 0 if material ID is not supported by encoding" && isValid);

    isValid &= commonSettings.cameraAttachedReflectionMaterialID != 0.0f || GetLibraryDesc().normalEncoding == NormalEncoding::R10_G10_B10_A2_UNORM;
    assert("'cameraAttachedReflectionMaterialID' can't be 0 if material ID is not supported by encoding" && isValid);

    memcpy(&m_CommonSettings, &commonSettings, sizeof(commonSettings));

    // Rotators (respecting sample patterns symmetry)
    float angle1 = Sequence::Weyl1D(0.5f, commonSettings.frameIndex) * radians(90.0f);
    m_RotatorPre = Geometry::GetRotator(angle1);

    float a0 = Sequence::Weyl1D(0.0f, commonSettings.frameIndex * 2) * radians(90.0f);
    float a1 = Sequence::Bayer4x4(uint2(0, 0), commonSettings.frameIndex * 2) * radians(360.0f);
    m_Rotator = Geometry::CombineRotators(Geometry::GetRotator(a0), Geometry::GetRotator(a1));

    float a2 = Sequence::Weyl1D(0.0f, commonSettings.frameIndex * 2 + 1) * radians(90.0f);
    float a3 = Sequence::Bayer4x4(uint2(0, 0), commonSettings.frameIndex * 2 + 1) * radians(360.0f);
    m_RotatorPost = Geometry::CombineRotators(Geometry::GetRotator(a2), Geometry::GetRotator(a3));

    // Main matrices
    m_ViewToClip = float4x4
    (
        float4(m_CommonSettings.viewToClipMatrix),
        float4(m_CommonSettings.viewToClipMatrix + 4),
        float4(m_CommonSettings.viewToClipMatrix + 8),
        float4(m_CommonSettings.viewToClipMatrix + 12)
    );

    m_ViewToClipPrev = float4x4
    (
        float4(m_CommonSettings.viewToClipMatrixPrev),
        float4(m_CommonSettings.viewToClipMatrixPrev + 4),
        float4(m_CommonSettings.viewToClipMatrixPrev + 8),
        float4(m_CommonSettings.viewToClipMatrixPrev + 12)
    );

    m_WorldToView = float4x4
    (
        float4(m_CommonSettings.worldToViewMatrix),
        float4(m_CommonSettings.worldToViewMatrix + 4),
        float4(m_CommonSettings.worldToViewMatrix + 8),
        float4(m_CommonSettings.worldToViewMatrix + 12)
    );

    m_WorldToViewPrev = float4x4
    (
        float4(m_CommonSettings.worldToViewMatrixPrev),
        float4(m_CommonSettings.worldToViewMatrixPrev + 4),
        float4(m_CommonSettings.worldToViewMatrixPrev + 8),
        float4(m_CommonSettings.worldToViewMatrixPrev + 12)
    );

    m_WorldPrevToWorld = float4x4
    (
        float4(m_CommonSettings.worldPrevToWorldMatrix),
        float4(m_CommonSettings.worldPrevToWorldMatrix + 4),
        float4(m_CommonSettings.worldPrevToWorldMatrix + 8),
        float4(m_CommonSettings.worldPrevToWorldMatrix + 12)
    );

    // There are many cases, where history buffers contain garbage - handle at least one of them internally
    if (m_IsFirstUse)
    {
        m_CommonSettings.accumulationMode = AccumulationMode::CLEAR_AND_RESTART;
        m_WorldToViewPrev = m_WorldToView;
        m_ViewToClipPrev = m_ViewToClip;
        m_IsFirstUse = false;
    }

    // Convert to LH
    uint32_t flags = 0;
    DecomposeProjection(STYLE_D3D, STYLE_D3D, m_ViewToClip, &flags, nullptr, nullptr, m_Frustum.a, nullptr, nullptr);

    if ( !(flags & PROJ_LEFT_HANDED) )
    {
        m_ViewToClip.col2 = -m_ViewToClip[2];
        m_ViewToClipPrev.col2 = -m_ViewToClipPrev[2];

        m_WorldToView.Transpose();
        m_WorldToView.col2 = -m_WorldToView[2];
        m_WorldToView.Transpose();

        m_WorldToViewPrev.Transpose();
        m_WorldToViewPrev.col2 = -m_WorldToViewPrev[2];
        m_WorldToViewPrev.Transpose();
    }

    // Compute other matrices
    m_ViewToWorld = m_WorldToView;
    m_ViewToWorld.InvertOrtho();

    m_ViewToWorldPrev = m_WorldToViewPrev;
    m_ViewToWorldPrev.InvertOrtho();

    const float3& cameraPosition = m_ViewToWorld[3].xyz;
    const float3& cameraPositionPrev = m_ViewToWorldPrev[3].xyz;
    float3 translationDelta = cameraPositionPrev - cameraPosition;

    // IMPORTANT: this part is mandatory needed to preserve precision by making matrices camera relative
    m_ViewToWorld.SetTranslation( float3::Zero() );
    m_WorldToView = m_ViewToWorld;
    m_WorldToView.InvertOrtho();

    m_ViewToWorldPrev.SetTranslation( translationDelta );
    m_WorldToViewPrev = m_ViewToWorldPrev;
    m_WorldToViewPrev.InvertOrtho();

    m_WorldToClip = m_ViewToClip * m_WorldToView;
    m_WorldToClipPrev = m_ViewToClipPrev * m_WorldToViewPrev;

    m_ClipToWorldPrev = m_WorldToClipPrev;
    m_ClipToWorldPrev.Invert();

    m_ClipToView = m_ViewToClip;
    m_ClipToView.Invert();

    m_ClipToViewPrev = m_ViewToClipPrev;
    m_ClipToViewPrev.Invert();

    m_ClipToWorld = m_WorldToClip;
    m_ClipToWorld.Invert();

    float project[3];
    float settings[PROJ_NUM];
    DecomposeProjection(STYLE_D3D, STYLE_D3D, m_ViewToClip, &flags, settings, nullptr, m_Frustum.a, project, nullptr);
    m_ProjectY = project[1];
    m_OrthoMode = (flags & PROJ_ORTHO) ? -1.0f : 0.0f;

    DecomposeProjection(STYLE_D3D, STYLE_D3D, m_ViewToClipPrev, &flags, nullptr, nullptr, m_FrustumPrev.a, nullptr, nullptr);

    m_ViewDirection = -float3(m_ViewToWorld[2]);
    m_ViewDirectionPrev = -float3(m_ViewToWorldPrev[2]);

    m_CameraDelta = float3(translationDelta.x, translationDelta.y, translationDelta.z);

    m_Timer.UpdateElapsedTimeSinceLastSave();
    m_Timer.SaveCurrentTime();

    m_TimeDelta = m_CommonSettings.timeDeltaBetweenFrames > 0.0f ? m_CommonSettings.timeDeltaBetweenFrames : m_Timer.GetSmoothedElapsedTime();
    m_FrameRateScale = max(33.333f / m_TimeDelta, 1.0f);

    float dx = abs(m_CommonSettings.cameraJitter[0] - m_CommonSettings.cameraJitterPrev[0]);
    float dy = abs(m_CommonSettings.cameraJitter[1] - m_CommonSettings.cameraJitterPrev[1]);
    m_JitterDelta = max(dx, dy);

    float FPS = m_FrameRateScale * 30.0f;
    float nonLinearAccumSpeed = FPS * 0.25f / (1.0f + FPS * 0.25f);
    m_CheckerboardResolveAccumSpeed = lerp(nonLinearAccumSpeed, 0.5f, m_JitterDelta);

    return isValid ? Result::SUCCESS : Result::INVALID_ARGUMENT;
}

nrd::Result nrd::InstanceImpl::SetDenoiserSettings(Identifier identifier, const void* denoiserSettings)
{
    for (DenoiserData& denoiserData : m_DenoiserData)
    {
        if (denoiserData.desc.identifier == identifier)
        {
            memcpy(&denoiserData.settings, denoiserSettings, denoiserData.settingsSize);

            return Result::SUCCESS;
        }
    }

    return Result::INVALID_ARGUMENT;
}

nrd::Result nrd::InstanceImpl::GetComputeDispatches(const Identifier* identifiers, uint32_t identifiersNum, const DispatchDesc*& dispatchDescs, uint32_t& dispatchDescsNum)
{
    m_ConstantDataOffset = 0;
    m_ActiveDispatches.clear();

    // Trivial checks
    if (!identifiers || !identifiersNum)
    {
        dispatchDescs = nullptr;
        dispatchDescsNum = 0;

        return !identifiersNum ? Result::SUCCESS : Result::INVALID_ARGUMENT;
    }

    // Inject "clear" calls if needed
    if (m_CommonSettings.accumulationMode == AccumulationMode::CLEAR_AND_RESTART)
    {
        for (const ClearResource& clearResource : m_ClearResources)
        {
            // If current denoiser is in list
            if (!IsInList(clearResource.identifier, identifiers, identifiersNum))
                continue;

            // Add a clear dispatch
            const InternalDispatchDesc& internalDispatchDesc = m_Dispatches[ m_DispatchClearIndex[clearResource.isInteger ? 1 : 0] ];

            uint16_t w = DivideUp(m_CommonSettings.resourceSize[0], clearResource.downsampleFactor);
            uint16_t h = DivideUp(m_CommonSettings.resourceSize[1], clearResource.downsampleFactor);

            DispatchDesc dispatchDesc = {};
            dispatchDesc.name = internalDispatchDesc.name;
            dispatchDesc.identifier = clearResource.identifier;
            dispatchDesc.resources = &clearResource.resource;
            dispatchDesc.resourcesNum = 1;
            dispatchDesc.pipelineIndex = internalDispatchDesc.pipelineIndex;
            dispatchDesc.gridWidth = DivideUp(w, internalDispatchDesc.numThreads.width);
            dispatchDesc.gridHeight = DivideUp(h, internalDispatchDesc.numThreads.height);

            m_ActiveDispatches.push_back(dispatchDesc);
        }
    }

    // Collect dispatches for requested denoisers
    for (const DenoiserData& denoiserData : m_DenoiserData)
    {
        // If current denoiser is in list
        if (!IsInList(denoiserData.desc.identifier, identifiers, identifiersNum))
            continue;

        // Update denoiser and gather dispatches
        UpdatePingPong(denoiserData);

        if (denoiserData.desc.denoiser == Denoiser::REBLUR_DIFFUSE || denoiserData.desc.denoiser == Denoiser::REBLUR_DIFFUSE_SH ||
            denoiserData.desc.denoiser == Denoiser::REBLUR_SPECULAR || denoiserData.desc.denoiser == Denoiser::REBLUR_SPECULAR_SH ||
            denoiserData.desc.denoiser == Denoiser::REBLUR_DIFFUSE_SPECULAR || denoiserData.desc.denoiser == Denoiser::REBLUR_DIFFUSE_SPECULAR_SH ||
            denoiserData.desc.denoiser == Denoiser::REBLUR_DIFFUSE_DIRECTIONAL_OCCLUSION)
            Update_Reblur(denoiserData);
        else if (denoiserData.desc.denoiser == Denoiser::REBLUR_DIFFUSE_OCCLUSION ||
            denoiserData.desc.denoiser == Denoiser::REBLUR_SPECULAR_OCCLUSION ||
            denoiserData.desc.denoiser == Denoiser::REBLUR_DIFFUSE_SPECULAR_OCCLUSION)
            Update_ReblurOcclusion(denoiserData);
        else if (denoiserData.desc.denoiser == Denoiser::RELAX_DIFFUSE || denoiserData.desc.denoiser == Denoiser::RELAX_DIFFUSE_SH ||
            denoiserData.desc.denoiser == Denoiser::RELAX_SPECULAR || denoiserData.desc.denoiser == Denoiser::RELAX_SPECULAR_SH ||
            denoiserData.desc.denoiser == Denoiser::RELAX_DIFFUSE_SPECULAR || denoiserData.desc.denoiser == Denoiser::RELAX_DIFFUSE_SPECULAR_SH)
            Update_Relax(denoiserData);
        else if (denoiserData.desc.denoiser == Denoiser::SIGMA_SHADOW || denoiserData.desc.denoiser == Denoiser::SIGMA_SHADOW_TRANSLUCENCY)
            Update_SigmaShadow(denoiserData);
        else if (denoiserData.desc.denoiser == Denoiser::REFERENCE)
            Update_Reference(denoiserData);
    }

    // Maximize CB reuse
    for (size_t i = 1; i < m_ActiveDispatches.size(); i++)
    {
        const DispatchDesc& dispatchDescPrev = m_ActiveDispatches[i - 1];
        DispatchDesc& dispatchDescCurr = m_ActiveDispatches[i];
        if (dispatchDescPrev.constantBufferDataSize == dispatchDescCurr.constantBufferDataSize)
        {
            if (!memcmp(dispatchDescPrev.constantBufferData, dispatchDescCurr.constantBufferData, dispatchDescCurr.constantBufferDataSize))
                dispatchDescCurr.constantBufferDataMatchesPreviousDispatch = true;
        }
    }

    // Output
    dispatchDescs = m_ActiveDispatches.data();
    dispatchDescsNum = (uint32_t)m_ActiveDispatches.size();

    return dispatchDescsNum ? Result::SUCCESS : Result::INVALID_ARGUMENT;
}

void nrd::InstanceImpl::AddComputeDispatchDesc
(
    NumThreads numThreads,
    uint16_t downsampleFactor,
    uint32_t constantBufferDataSize,
    uint32_t maxRepeatNum,
    const char* shaderFileName,
    const ComputeShaderDesc& dxbc,
    const ComputeShaderDesc& dxil,
    const ComputeShaderDesc& spirv
)
{
    // Pipeline (unique only)
    size_t pipelineIndex = 0;
    for (; pipelineIndex < m_Pipelines.size(); pipelineIndex++)
    {
        const PipelineDesc& pipeline = m_Pipelines[pipelineIndex];

        if (!strcmp(pipeline.shaderFileName, shaderFileName))
            break;
    }

    if (pipelineIndex == m_Pipelines.size())
    {
        PipelineDesc pipelineDesc = {};
        pipelineDesc.shaderFileName = shaderFileName;
        pipelineDesc.shaderEntryPointName = NRD_STRINGIFY(NRD_CS_MAIN);
        pipelineDesc.computeShaderDXBC = dxbc;
        pipelineDesc.computeShaderDXIL = dxil;
        pipelineDesc.computeShaderSPIRV = spirv;
        pipelineDesc.resourceRanges = (ResourceRangeDesc*)m_ResourceRanges.size();
        pipelineDesc.hasConstantData = constantBufferDataSize != 0;

        for (size_t r = 0; r < 2; r++)
        {
            ResourceRangeDesc descriptorRange = {};
            descriptorRange.descriptorType = r == 0 ? DescriptorType::TEXTURE : DescriptorType::STORAGE_TEXTURE;

            for (size_t i = m_ResourceOffset; i < m_Resources.size(); i++ )
            {
                const ResourceDesc& resource = m_Resources[i];
                if (descriptorRange.descriptorType == resource.descriptorType)
                    descriptorRange.descriptorsNum++;
            }

            if (descriptorRange.descriptorsNum != 0)
            {
                m_ResourceRanges.push_back(descriptorRange);
                pipelineDesc.resourceRangesNum++;
            }
        }

        m_Pipelines.push_back( pipelineDesc );
    }

    // Dispatch
    InternalDispatchDesc computeDispatchDesc = {};
    computeDispatchDesc.name = m_PassName;
    computeDispatchDesc.pipelineIndex = (uint16_t)pipelineIndex;
    computeDispatchDesc.downsampleFactor = downsampleFactor;
    computeDispatchDesc.maxRepeatsNum = (uint16_t)maxRepeatNum;
    computeDispatchDesc.constantBufferDataSize = constantBufferDataSize;
    computeDispatchDesc.resourcesNum = uint32_t(m_Resources.size() - m_ResourceOffset);
    computeDispatchDesc.resources = (ResourceDesc*)m_ResourceOffset;
    computeDispatchDesc.numThreads = numThreads;

    m_Dispatches.push_back(computeDispatchDesc);
}

void nrd::InstanceImpl::PrepareDesc()
{
    m_Desc = {};

    m_Desc.constantBufferRegisterIndex = NRD_CONSTANT_BUFFER_REGISTER_INDEX;
    m_Desc.constantBufferSpaceIndex = NRD_CONSTANT_BUFFER_SPACE_INDEX;

    m_Desc.samplers = g_Samplers.data();
    m_Desc.samplersNum = (uint32_t)g_Samplers.size();
    m_Desc.samplersSpaceIndex = NRD_SAMPLERS_SPACE_INDEX;
    m_Desc.samplersBaseRegisterIndex = 0;

    m_Desc.pipelines = m_Pipelines.data();
    m_Desc.pipelinesNum = (uint32_t)m_Pipelines.size();
    m_Desc.resourcesSpaceIndex = NRD_RESOURCES_SPACE_INDEX;

    m_Desc.permanentPool = m_PermanentPool.data();
    m_Desc.permanentPoolSize = (uint32_t)m_PermanentPool.size();

    m_Desc.transientPool = m_TransientPool.data();
    m_Desc.transientPoolSize = (uint32_t)m_TransientPool.size();

    const bool samplersAreInSeparateSet = NRD_SAMPLERS_SPACE_INDEX != NRD_CONSTANT_BUFFER_SPACE_INDEX && NRD_SAMPLERS_SPACE_INDEX != NRD_RESOURCES_SPACE_INDEX;
    if (samplersAreInSeparateSet)
        m_Desc.descriptorPoolDesc.samplersMaxNum += m_Desc.samplersNum;

    // Calculate descriptor heap (sets) requirements
    for (InternalDispatchDesc& dispatchDesc : m_Dispatches)
    {
        size_t textureOffset = (size_t)dispatchDesc.resources;
        dispatchDesc.resources = &m_Resources[textureOffset];

        for (uint32_t i = 0; i < dispatchDesc.resourcesNum; i++)
        {
            const ResourceDesc& resource = dispatchDesc.resources[i];
            if (resource.descriptorType == DescriptorType::TEXTURE)
                m_Desc.descriptorPoolDesc.texturesMaxNum += dispatchDesc.maxRepeatsNum;
            else if (resource.descriptorType == DescriptorType::STORAGE_TEXTURE)
                m_Desc.descriptorPoolDesc.storageTexturesMaxNum += dispatchDesc.maxRepeatsNum;
        }

        m_Desc.descriptorPoolDesc.setsMaxNum += dispatchDesc.maxRepeatsNum;

        if (!samplersAreInSeparateSet)
            m_Desc.descriptorPoolDesc.samplersMaxNum += dispatchDesc.maxRepeatsNum * m_Desc.samplersNum;

        if (dispatchDesc.constantBufferDataSize != 0)
        {
            m_Desc.descriptorPoolDesc.constantBuffersMaxNum += dispatchDesc.maxRepeatsNum;
            m_Desc.constantBufferMaxDataSize = max(dispatchDesc.constantBufferDataSize, m_Desc.constantBufferMaxDataSize);
        }
    }

    // For potential clears
    uint32_t clearNum = (uint32_t)m_ClearResources.size();
    m_Desc.descriptorPoolDesc.storageTexturesMaxNum += clearNum;
    m_Desc.descriptorPoolDesc.setsMaxNum += clearNum;

    if (!samplersAreInSeparateSet)
        m_Desc.descriptorPoolDesc.samplersMaxNum += clearNum * m_Desc.samplersNum;

    // Assign resources
    for (PipelineDesc& pipelineDesc : m_Pipelines)
    {
        size_t descriptorRangeffset = (size_t)pipelineDesc.resourceRanges;
        pipelineDesc.resourceRanges = &m_ResourceRanges[descriptorRangeffset];
    }

    // *= number of "spaces"
    uint32_t descriptorSetNum = 1;
    if (m_Desc.constantBufferSpaceIndex != m_Desc.samplersSpaceIndex)
        descriptorSetNum++;
    if (m_Desc.samplersSpaceIndex != m_Desc.resourcesSpaceIndex)
        descriptorSetNum++;

    m_Desc.descriptorPoolDesc.setsMaxNum *= descriptorSetNum;
}

void nrd::InstanceImpl::UpdatePingPong(const DenoiserData& denoiserData)
{
    for (uint32_t i = 0; i < denoiserData.pingPongNum; i++)
    {
        PingPong& pingPong = m_PingPongs[denoiserData.pingPongOffset + i];
        ResourceDesc& resource = m_Resources[pingPong.resourceIndex];

        Swap(resource.indexInPool, pingPong.indexInPoolToSwapWith);
    }
}

void nrd::InstanceImpl::PushTexture(DescriptorType descriptorType, uint16_t localIndex, uint16_t indexToSwapWith)
{
    ResourceType resourceType = (ResourceType)localIndex;
    uint16_t globalIndex = 0;

    if (localIndex >= TRANSIENT_POOL_START)
    {
        resourceType = ResourceType::TRANSIENT_POOL;
        globalIndex = m_IndexRemap[localIndex - TRANSIENT_POOL_START];

        if (indexToSwapWith != uint16_t(-1))
        {
            assert(indexToSwapWith >= TRANSIENT_POOL_START && indexToSwapWith < PERMANENT_POOL_START);

            indexToSwapWith = m_IndexRemap[indexToSwapWith - TRANSIENT_POOL_START];
            m_PingPongs.push_back( {m_Resources.size(), indexToSwapWith} );
        }
    }
    else if (localIndex >= PERMANENT_POOL_START)
    {
        resourceType = ResourceType::PERMANENT_POOL;
        globalIndex = m_PermanentPoolOffset + localIndex - PERMANENT_POOL_START;

        if (indexToSwapWith != uint16_t(-1))
        {
            assert(indexToSwapWith >= PERMANENT_POOL_START);

            indexToSwapWith = m_PermanentPoolOffset + indexToSwapWith - PERMANENT_POOL_START;
            m_PingPongs.push_back( {m_Resources.size(), indexToSwapWith} );
        }
    }

    m_Resources.push_back( {descriptorType, resourceType, globalIndex} );
}

void nrd::InstanceImpl::AddTextureToTransientPool(const TextureDesc& textureDesc)
{
    // Try to find a replacement from previous denoisers
    for (uint16_t i = 0; i < m_TransientPoolOffset; i++)
    {
        // Format and dimensions must match
        const TextureDesc& t = m_TransientPool[i];
        if (t.format == textureDesc.format && t.downsampleFactor == textureDesc.downsampleFactor)
        {
            // The candidate must not be already in use in the current denoiser
            size_t j = 0;
            for (; j < m_IndexRemap.size(); j++)
            {
                if (m_IndexRemap[j] == i)
                    break;
            }

            // A replacement is found - reuse memory
            if (j == m_IndexRemap.size())
            {
                m_IndexRemap.push_back(i);

                return;
            }
        }
    }

    // A replacement is not found - add memory
    m_IndexRemap.push_back( (uint16_t)m_TransientPool.size() );
    m_TransientPool.push_back(textureDesc);
}

void* nrd::InstanceImpl::PushDispatch(const DenoiserData& denoiserData, uint32_t localIndex)
{
    size_t dispatchIndex = denoiserData.dispatchOffset + localIndex;
    const InternalDispatchDesc& internalDispatchDesc = m_Dispatches[dispatchIndex];

    // Copy data
    DispatchDesc dispatchDesc = {};
    dispatchDesc.name = internalDispatchDesc.name;
    dispatchDesc.identifier = internalDispatchDesc.identifier;
    dispatchDesc.resources = internalDispatchDesc.resources;
    dispatchDesc.resourcesNum = internalDispatchDesc.resourcesNum;
    dispatchDesc.pipelineIndex = internalDispatchDesc.pipelineIndex;

    // Update constant data
    if (m_ConstantDataOffset + internalDispatchDesc.constantBufferDataSize > CONSTANT_DATA_SIZE)
    {
        assert("Constant data doesn't fit into the prealocated array!" && false);
        dispatchDesc.constantBufferData = nullptr; // TODO: better crash
    }
    else
        dispatchDesc.constantBufferData = m_ConstantData + m_ConstantDataOffset;

    dispatchDesc.constantBufferDataSize = internalDispatchDesc.constantBufferDataSize;
    m_ConstantDataOffset += internalDispatchDesc.constantBufferDataSize;

    // Needed for "constantBufferDataMatchesPreviousDispatch"
    if (dispatchDesc.constantBufferData)
        memset((void*)dispatchDesc.constantBufferData, 0, dispatchDesc.constantBufferDataSize);

    // Update grid size
    uint16_t w = m_CommonSettings.rectSize[0];
    uint16_t h = m_CommonSettings.rectSize[1];
    uint16_t d = internalDispatchDesc.downsampleFactor;

    if (d == USE_MAX_DIMS)
    {
        w = max(w, m_CommonSettings.rectSizePrev[0]);
        h = max(h, m_CommonSettings.rectSizePrev[1]);
        d = 1;
    }
    else if (d == IGNORE_RS)
    {
        w = m_CommonSettings.resourceSize[0];
        h = m_CommonSettings.resourceSize[1];
        d = 1;
    }

    w = DivideUp(w, d);
    h = DivideUp(h, d);

    dispatchDesc.gridWidth = DivideUp(w, internalDispatchDesc.numThreads.width);
    dispatchDesc.gridHeight = DivideUp(h, internalDispatchDesc.numThreads.height);

    // Store
    m_ActiveDispatches.push_back(dispatchDesc);

    return (void*)dispatchDesc.constantBufferData;
}
