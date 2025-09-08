/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "InstanceImpl.h"

#include "../Shaders/Include/SIGMA_Config.hlsli"
#include "../Shaders/SIGMA_Blur.resources.hlsli"
#include "../Shaders/SIGMA_ClassifyTiles.resources.hlsli"
#include "../Shaders/SIGMA_Copy.resources.hlsli"
#include "../Shaders/SIGMA_SmoothTiles.resources.hlsli"
#include "../Shaders/SIGMA_SplitScreen.resources.hlsli"
#include "../Shaders/SIGMA_TemporalStabilization.resources.hlsli"

// Permutations
#define SIGMA_POST_BLUR_PERMUTATION_NUM 2
#define SIGMA_NO_PERMUTATIONS           1

void nrd::InstanceImpl::Update_SigmaShadow(const DenoiserData& denoiserData) {
    enum class Dispatch {
        CLASSIFY_TILES,
        SMOOTH_TILES = CLASSIFY_TILES + SIGMA_NO_PERMUTATIONS,
        COPY = SMOOTH_TILES + SIGMA_NO_PERMUTATIONS,
        BLUR = COPY + SIGMA_NO_PERMUTATIONS,
        POST_BLUR = BLUR + SIGMA_NO_PERMUTATIONS,
        TEMPORAL_STABILIZATION = POST_BLUR + SIGMA_POST_BLUR_PERMUTATION_NUM,
        SPLIT_SCREEN = TEMPORAL_STABILIZATION + SIGMA_NO_PERMUTATIONS,
    };

    const SigmaSettings& settings = denoiserData.settings.sigma;

    // SPLIT_SCREEN (passthrough)
    if (m_CommonSettings.splitScreen >= 1.0f) {
        void* consts = PushDispatch(denoiserData, AsUint(Dispatch::SPLIT_SCREEN));
        AddSharedConstants_Sigma(settings, consts);

        return;
    }

    { // CLASSIFY_TILES
        void* consts = PushDispatch(denoiserData, AsUint(Dispatch::CLASSIFY_TILES));
        AddSharedConstants_Sigma(settings, consts);
    }

    { // SMOOTH_TILES
        void* consts = PushDispatch(denoiserData, AsUint(Dispatch::SMOOTH_TILES));
        AddSharedConstants_Sigma(settings, consts);
    }

    // COPY
    if (settings.maxStabilizedFrameNum) {
        void* consts = PushDispatch(denoiserData, AsUint(Dispatch::COPY));
        AddSharedConstants_Sigma(settings, consts);
    }

    { // BLUR
        void* consts = PushDispatch(denoiserData, AsUint(Dispatch::BLUR));
        AddSharedConstants_Sigma(settings, consts);
    }

    { // POST_BLUR
        uint32_t passIndex = AsUint(Dispatch::POST_BLUR) + (settings.maxStabilizedFrameNum ? 1 : 0);
        void* consts = PushDispatch(denoiserData, passIndex);
        AddSharedConstants_Sigma(settings, consts);
    }

    // TEMPORAL_STABILIZATION
    if (settings.maxStabilizedFrameNum) {
        void* consts = PushDispatch(denoiserData, AsUint(Dispatch::TEMPORAL_STABILIZATION));
        AddSharedConstants_Sigma(settings, consts);
    }

    // SPLIT_SCREEN
    if (m_CommonSettings.splitScreen > 0.0f) {
        void* consts = PushDispatch(denoiserData, AsUint(Dispatch::SPLIT_SCREEN));
        AddSharedConstants_Sigma(settings, consts);
    }
}

void nrd::InstanceImpl::AddSharedConstants_Sigma(const SigmaSettings& settings, void* data) {
    struct SharedConstants {
        SIGMA_SHARED_CONSTANTS
    };

    NRD_DECLARE_DIMS;

    float unproject = 1.0f / (0.5f * rectH * m_ProjectY);
    uint16_t tilesW = DivideUp(rectW, 16);
    uint16_t tilesH = DivideUp(rectH, 16);

    bool isRectChanged = rectW != rectWprev || rectH != rectHprev;
    uint32_t frameNum = min(settings.maxStabilizedFrameNum, SIGMA_MAX_HISTORY_FRAME_NUM);
    float3 lightDirectionView = Rotate(m_WorldToView, float3(settings.lightDirection[0], settings.lightDirection[1], settings.lightDirection[2]));
    float stabilizationStrength = frameNum / (1.0f + frameNum);

    SharedConstants* consts = (SharedConstants*)data;
    consts->gWorldToView = m_WorldToView;
    consts->gViewToClip = m_ViewToClip;
    consts->gWorldToClipPrev = m_WorldToClipPrev;
    consts->gWorldToViewPrev = m_WorldToViewPrev;
    consts->gRotator = m_Rotator;
    consts->gRotatorPost = m_RotatorPost;
    consts->gViewVectorWorld = m_ViewDirection.xmm;
    consts->gLightDirectionView = float4(lightDirectionView.x, lightDirectionView.y, lightDirectionView.z, 0.0f);
    consts->gFrustum = m_Frustum;
    consts->gFrustumPrev = m_FrustumPrev;
    consts->gCameraDelta = m_CameraDelta.xmm;
    consts->gMvScale = float4(m_CommonSettings.motionVectorScale[0], m_CommonSettings.motionVectorScale[1], m_CommonSettings.motionVectorScale[2], m_CommonSettings.isMotionVectorInWorldSpace ? 1.0f : 0.0f);
    consts->gResourceSizeInv = float2(1.0f / float(resourceW), 1.0f / float(resourceH));
    consts->gResourceSizeInvPrev = float2(1.0f / float(resourceWprev), 1.0f / float(resourceHprev));
    consts->gRectSize = float2(float(rectW), float(rectH));
    consts->gRectSizeInv = float2(1.0f / float(rectW), 1.0f / float(rectH));
    consts->gRectSizePrev = float2(float(rectWprev), float(rectHprev));
    consts->gResolutionScale = float2(float(rectW) / float(resourceW), float(rectH) / float(resourceH));
    consts->gRectOffset = float2(float(m_CommonSettings.rectOrigin[0]) / float(resourceW), float(m_CommonSettings.rectOrigin[1]) / float(resourceH));
    consts->gPrintfAt = uint2(m_CommonSettings.printfAt[0], m_CommonSettings.printfAt[1]);
    consts->gRectOrigin = uint2(m_CommonSettings.rectOrigin[0], m_CommonSettings.rectOrigin[1]);
    consts->gRectSizeMinusOne = int2(rectW - 1, rectH - 1);
    consts->gTilesSizeMinusOne = int2(tilesW - 1, tilesH - 1);
    consts->gOrthoMode = m_OrthoMode;
    consts->gUnproject = unproject;
    consts->gDenoisingRange = m_CommonSettings.denoisingRange;
    consts->gPlaneDistSensitivity = settings.planeDistanceSensitivity;
    consts->gStabilizationStrength = m_CommonSettings.accumulationMode == AccumulationMode::CONTINUE ? stabilizationStrength : 0.0f;
    consts->gDebug = m_CommonSettings.debug;
    consts->gSplitScreen = m_CommonSettings.splitScreen;
    consts->gViewZScale = m_CommonSettings.viewZScale;
    consts->gMinRectDimMulUnproject = (float)min(rectW, rectH) * unproject;
    consts->gFrameIndex = m_CommonSettings.frameIndex;
    consts->gIsRectChanged = isRectChanged ? 1 : 0;
}

// Shaders
#if NRD_EMBEDS_DXBC_SHADERS
#    include "SIGMA_ClassifyTiles.cs.dxbc.h"
#    include "SIGMA_SmoothTiles.cs.dxbc.h"
#    include "SIGMA_Copy.cs.dxbc.h"
#    include "SIGMA_Blur.cs.dxbc.h"
#    include "SIGMA_TemporalStabilization.cs.dxbc.h"
#    include "SIGMA_SplitScreen.cs.dxbc.h"
#endif

#if NRD_EMBEDS_DXIL_SHADERS
#    include "SIGMA_ClassifyTiles.cs.dxil.h"
#    include "SIGMA_SmoothTiles.cs.dxil.h"
#    include "SIGMA_Copy.cs.dxil.h"
#    include "SIGMA_Blur.cs.dxil.h"
#    include "SIGMA_TemporalStabilization.cs.dxil.h"
#    include "SIGMA_SplitScreen.cs.dxil.h"
#endif

#if NRD_EMBEDS_SPIRV_SHADERS
#    include "SIGMA_ClassifyTiles.cs.spirv.h"
#    include "SIGMA_SmoothTiles.cs.spirv.h"
#    include "SIGMA_Copy.cs.spirv.h"
#    include "SIGMA_Blur.cs.spirv.h"
#    include "SIGMA_TemporalStabilization.cs.spirv.h"
#    include "SIGMA_SplitScreen.cs.spirv.h"
#endif

#include "Denoisers/Sigma_Shadow.hpp"
#include "Denoisers/Sigma_ShadowTranslucency.hpp"
