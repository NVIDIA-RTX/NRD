/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "InstanceImpl.h"

#include "../Shaders/Include/RELAX_Config.hlsli"
#include "../Shaders/Resources/RELAX_AntiFirefly.resources.hlsli"
#include "../Shaders/Resources/RELAX_Atrous.resources.hlsli"
#include "../Shaders/Resources/RELAX_AtrousSmem.resources.hlsli"
#include "../Shaders/Resources/RELAX_ClassifyTiles.resources.hlsli"
#include "../Shaders/Resources/RELAX_Copy.resources.hlsli"
#include "../Shaders/Resources/RELAX_HistoryClamping.resources.hlsli"
#include "../Shaders/Resources/RELAX_HistoryFix.resources.hlsli"
#include "../Shaders/Resources/RELAX_HitDistReconstruction.resources.hlsli"
#include "../Shaders/Resources/RELAX_PrePass.resources.hlsli"
#include "../Shaders/Resources/RELAX_SplitScreen.resources.hlsli"
#include "../Shaders/Resources/RELAX_TemporalAccumulation.resources.hlsli"
#include "../Shaders/Resources/RELAX_Validation.resources.hlsli"

// Permutations
#define RELAX_HITDIST_RECONSTRUCTION_PERMUTATION_NUM 2
#define RELAX_PREPASS_PERMUTATION_NUM                2
#define RELAX_TEMPORAL_ACCUMULATION_PERMUTATION_NUM  4
#define RELAX_ATROUS_PERMUTATION_NUM                 2 // * RELAX_ATROUS_BINDING_VARIANT_NUM

// Other
#define RELAX_DUMMY                      AsUint(ResourceType::IN_VIEWZ)
#define RELAX_NO_PERMUTATIONS            1
#define RELAX_ATROUS_BINDING_VARIANT_NUM 5

constexpr uint32_t RELAX_MAX_ATROUS_PASS_NUM = 8;

#define RELAX_ADD_VALIDATION_DISPATCH \
    PushPass("Validation"); \
    { \
        PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS)); \
        PushInput(AsUint(ResourceType::IN_VIEWZ)); \
        PushInput(AsUint(ResourceType::IN_MV)); \
        PushInput(AsUint(Transient::HISTORY_LENGTH)); \
        PushOutput(AsUint(ResourceType::OUT_VALIDATION)); \
        AddDispatch(RELAX_Validation, RELAX_Validation, IGNORE_RS); \
    }

inline float3 RELAX_GetFrustumForward(const float4x4& viewToWorld, const float4& frustum) {
    float4 frustumForwardView = float4(0.5f, 0.5f, 1.0f, 0.0f) * float4(frustum.z, frustum.w, 1.0f, 0.0f) + float4(frustum.x, frustum.y, 0.0f, 0.0f);
    float3 frustumForwardWorld = (viewToWorld * frustumForwardView).xyz;

    // Vector is not normalized for non-symmetric projections, it has to have .z = 1.0 to correctly reconstruct world position in shaders
    return frustumForwardWorld;
}

void nrd::InstanceImpl::AddSharedConstants_Relax(const RelaxSettings& settings, void* data) {
    struct SharedConstants {
        RELAX_SHARED_CONSTANTS
    };

    NRD_DECLARE_DIMS;

    float tanHalfFov = 1.0f / m_ViewToClip.a00;
    float aspect = m_ViewToClip.a00 / m_ViewToClip.a11;
    float3 frustumRight = m_WorldToView.Row(0).xyz * tanHalfFov;
    float3 frustumUp = m_WorldToView.Row(1).xyz * tanHalfFov * aspect;
    float3 frustumForward = RELAX_GetFrustumForward(m_ViewToWorld, m_Frustum);

    float prevTanHalfFov = 1.0f / m_ViewToClipPrev.a00;
    float prevAspect = m_ViewToClipPrev.a00 / m_ViewToClipPrev.a11;
    float3 prevFrustumRight = m_WorldToViewPrev.Row(0).xyz * prevTanHalfFov;
    float3 prevFrustumUp = m_WorldToViewPrev.Row(1).xyz * prevTanHalfFov * prevAspect;
    float3 prevFrustumForward = RELAX_GetFrustumForward(m_ViewToWorldPrev, m_FrustumPrev);

    float maxDiffuseLuminanceRelativeDifference = -log(saturate(settings.diffuseMinLuminanceWeight));
    float maxSpecularLuminanceRelativeDifference = -log(saturate(settings.specularMinLuminanceWeight));
    float disocclusionThresholdBonus = (1.0f + m_JitterDelta) / float(rectH);
    bool isHistoryReset = m_CommonSettings.accumulationMode != AccumulationMode::CONTINUE;

    // Checkerboard logic
    uint32_t specCheckerboard = 2;
    uint32_t diffCheckerboard = 2;
    switch (settings.checkerboardMode) {
        case CheckerboardMode::BLACK:
            diffCheckerboard = 0;
            specCheckerboard = 1;
            break;
        case CheckerboardMode::WHITE:
            diffCheckerboard = 1;
            specCheckerboard = 0;
            break;
    }

    SharedConstants* consts = (SharedConstants*)data;
    consts->gWorldToClip = m_WorldToClip;
    consts->gWorldToClipPrev = m_WorldToClipPrev;
    consts->gWorldToViewPrev = m_WorldToViewPrev;
    consts->gWorldPrevToWorld = m_WorldPrevToWorld;
    consts->gRotatorPre = m_RotatorPre;
    consts->gFrustumRight = float4(frustumRight, 0.0f);
    consts->gFrustumUp = float4(frustumUp, 0.0f);
    consts->gFrustumForward = float4(frustumForward, 0.0f);
    consts->gPrevFrustumRight = float4(prevFrustumRight, 0.0f);
    consts->gPrevFrustumUp = float4(prevFrustumUp, 0.0f);
    consts->gPrevFrustumForward = float4(prevFrustumForward, 0.0f);
    consts->gCameraDelta = float4(m_CameraDelta, 0.0f);
    consts->gMvScale = float4(m_CommonSettings.motionVectorScale[0], m_CommonSettings.motionVectorScale[1], m_CommonSettings.motionVectorScale[2], m_CommonSettings.isMotionVectorInWorldSpace ? 1.0f : 0.0f);
    consts->gJitter = float2(m_CommonSettings.cameraJitter[0], m_CommonSettings.cameraJitter[1]);
    consts->gResolutionScale = float2(float(rectW) / float(resourceW), float(rectH) / float(resourceH));
    consts->gRectOffset = float2(float(m_CommonSettings.rectOrigin[0]) / float(resourceW), float(m_CommonSettings.rectOrigin[1]) / float(resourceH));
    consts->gResourceSizeInv = float2(1.0f / resourceW, 1.0f / resourceH);
    consts->gResourceSize = float2(resourceW, resourceH);
    consts->gRectSizeInv = float2(1.0f / rectW, 1.0f / rectH);
    consts->gRectSizePrev = float2(float(rectWprev), float(rectHprev));
    consts->gResourceSizeInvPrev = float2(1.0f / resourceWprev, 1.0f / resourceHprev);
    consts->gPrintfAt = uint2(m_CommonSettings.printfAt[0], m_CommonSettings.printfAt[1]);
    consts->gRectOrigin = uint2(m_CommonSettings.rectOrigin[0], m_CommonSettings.rectOrigin[1]);
    consts->gRectSize = int2(rectW, rectH);
    consts->gSpecMaxAccumulatedFrameNum = isHistoryReset ? 0.0f : (float)min(settings.specularMaxAccumulatedFrameNum, RELAX_MAX_HISTORY_FRAME_NUM);
    consts->gSpecMaxFastAccumulatedFrameNum = isHistoryReset ? 0.0f : (float)min(settings.specularMaxFastAccumulatedFrameNum, RELAX_MAX_HISTORY_FRAME_NUM);
    consts->gDiffMaxAccumulatedFrameNum = isHistoryReset ? 0.0f : (float)min(settings.diffuseMaxAccumulatedFrameNum, RELAX_MAX_HISTORY_FRAME_NUM);
    consts->gDiffMaxFastAccumulatedFrameNum = isHistoryReset ? 0.0f : (float)min(settings.diffuseMaxFastAccumulatedFrameNum, RELAX_MAX_HISTORY_FRAME_NUM);
    consts->gDisocclusionThreshold = m_CommonSettings.disocclusionThreshold + disocclusionThresholdBonus;
    consts->gDisocclusionThresholdAlternate = m_CommonSettings.disocclusionThresholdAlternate + disocclusionThresholdBonus;
    consts->gCameraAttachedReflectionMaterialID = m_CommonSettings.cameraAttachedReflectionMaterialID;
    consts->gStrandMaterialID = m_CommonSettings.strandMaterialID;
    consts->gStrandThickness = m_CommonSettings.strandThickness;
    consts->gRoughnessFraction = settings.roughnessFraction;
    consts->gSpecVarianceBoost = settings.specularVarianceBoost;
    consts->gSplitScreen = m_CommonSettings.splitScreen;
    consts->gDiffBlurRadius = settings.diffusePrepassBlurRadius;
    consts->gSpecBlurRadius = settings.specularPrepassBlurRadius;
    consts->gDepthThreshold = settings.depthThreshold;
    consts->gLobeAngleFraction = settings.lobeAngleFraction;
    consts->gSpecLobeAngleSlack = radians(settings.specularLobeAngleSlack);
    consts->gHistoryFixEdgeStoppingNormalPower = settings.historyFixEdgeStoppingNormalPower;
    consts->gRoughnessEdgeStoppingRelaxation = settings.roughnessEdgeStoppingRelaxation;
    consts->gNormalEdgeStoppingRelaxation = settings.normalEdgeStoppingRelaxation;
    consts->gColorBoxSigmaScale = settings.historyClampingColorBoxSigmaScale;
    consts->gHistoryAccelerationAmount = settings.antilagSettings.accelerationAmount;
    consts->gHistoryResetTemporalSigmaScale = settings.antilagSettings.temporalSigmaScale;
    consts->gHistoryResetSpatialSigmaScale = settings.antilagSettings.spatialSigmaScale;
    consts->gHistoryResetAmount = settings.antilagSettings.resetAmount;
    consts->gDenoisingRange = m_CommonSettings.denoisingRange;
    consts->gSpecPhiLuminance = settings.specularPhiLuminance;
    consts->gDiffPhiLuminance = settings.diffusePhiLuminance;
    consts->gDiffMaxLuminanceRelativeDifference = maxDiffuseLuminanceRelativeDifference;
    consts->gSpecMaxLuminanceRelativeDifference = maxSpecularLuminanceRelativeDifference;
    consts->gLuminanceEdgeStoppingRelaxation = settings.roughnessEdgeStoppingRelaxation;
    consts->gConfidenceDrivenRelaxationMultiplier = settings.confidenceDrivenRelaxationMultiplier;
    consts->gConfidenceDrivenLuminanceEdgeStoppingRelaxation = settings.confidenceDrivenLuminanceEdgeStoppingRelaxation;
    consts->gConfidenceDrivenNormalEdgeStoppingRelaxation = settings.confidenceDrivenNormalEdgeStoppingRelaxation;
    consts->gDebug = m_CommonSettings.debug;
    consts->gOrthoMode = m_OrthoMode;
    consts->gUnproject = 1.0f / (0.5f * rectH * m_ProjectY);
    consts->gFramerateScale = clamp(16.66f / m_TimeDelta, 0.25f, 4.0f); // TODO: use m_FrameRateScale?
    consts->gCheckerboardResolveAccumSpeed = m_CheckerboardResolveAccumSpeed;
    consts->gJitterDelta = m_JitterDelta;
    consts->gHistoryFixFrameNum = settings.historyFixFrameNum + 1.0f;
    consts->gHistoryFixBasePixelStride = (float)settings.historyFixBasePixelStride;
    consts->gHistoryThreshold = (float)settings.spatialVarianceEstimationHistoryThreshold;
    consts->gViewZScale = m_CommonSettings.viewZScale;
    consts->gMinHitDistanceWeight = settings.minHitDistanceWeight * 2.0f; // TODO: 2 to match REBLUR units and make Pre passes identical (matches old default)
    consts->gDiffMinMaterial = settings.minMaterialForDiffuse;
    consts->gSpecMinMaterial = settings.minMaterialForSpecular;
    consts->gRoughnessEdgeStoppingEnabled = settings.enableRoughnessEdgeStopping ? 1 : 0;
    consts->gFrameIndex = m_CommonSettings.frameIndex;
    consts->gDiffCheckerboard = diffCheckerboard;
    consts->gSpecCheckerboard = specCheckerboard;
    consts->gHasHistoryConfidence = m_CommonSettings.isHistoryConfidenceAvailable ? 1 : 0;
    consts->gHasDisocclusionThresholdMix = m_CommonSettings.isDisocclusionThresholdMixAvailable ? 1 : 0;
    consts->gResetHistory = isHistoryReset ? 1 : 0;
}

void nrd::InstanceImpl::Update_Relax(const DenoiserData& denoiserData) {
    enum class Dispatch {
        CLASSIFY_TILES,
        HITDIST_RECONSTRUCTION = CLASSIFY_TILES + RELAX_NO_PERMUTATIONS,
        PREPASS = HITDIST_RECONSTRUCTION + RELAX_HITDIST_RECONSTRUCTION_PERMUTATION_NUM,
        TEMPORAL_ACCUMULATION = PREPASS + RELAX_PREPASS_PERMUTATION_NUM,
        HISTORY_FIX = TEMPORAL_ACCUMULATION + RELAX_TEMPORAL_ACCUMULATION_PERMUTATION_NUM,
        HISTORY_CLAMPING = HISTORY_FIX + RELAX_NO_PERMUTATIONS,
        COPY = HISTORY_CLAMPING + RELAX_NO_PERMUTATIONS,
        ANTI_FIREFLY = COPY + RELAX_NO_PERMUTATIONS,
        ATROUS = ANTI_FIREFLY + RELAX_NO_PERMUTATIONS,
        SPLIT_SCREEN = ATROUS + RELAX_ATROUS_PERMUTATION_NUM * RELAX_ATROUS_BINDING_VARIANT_NUM,
        VALIDATION = SPLIT_SCREEN + RELAX_NO_PERMUTATIONS,
    };

    NRD_DECLARE_DIMS;

    const RelaxSettings& settings = denoiserData.settings.relax;
    bool enableHitDistanceReconstruction = settings.hitDistanceReconstructionMode != HitDistanceReconstructionMode::OFF && settings.checkerboardMode == CheckerboardMode::OFF;
    uint32_t iterationNum = clamp(settings.atrousIterationNum, 2u, RELAX_MAX_ATROUS_PASS_NUM);

    // SPLIT_SCREEN (passthrough)
    if (m_CommonSettings.splitScreen >= 1.0f) {
        void* consts = PushDispatch(denoiserData, AsUint(Dispatch::SPLIT_SCREEN));
        AddSharedConstants_Relax(settings, consts);

        return;
    }

    { // CLASSIFY_TILES
        void* consts = PushDispatch(denoiserData, AsUint(Dispatch::CLASSIFY_TILES));
        AddSharedConstants_Relax(settings, consts);
    }

    // HITDIST_RECONSTRUCTION
    if (enableHitDistanceReconstruction) {
        bool is5x5 = settings.hitDistanceReconstructionMode == HitDistanceReconstructionMode::AREA_5X5;
        uint32_t passIndex = AsUint(Dispatch::HITDIST_RECONSTRUCTION) + (is5x5 ? 1 : 0);
        void* consts = PushDispatch(denoiserData, passIndex);
        AddSharedConstants_Relax(settings, consts);
    }

    { // PREPASS
        uint32_t passIndex = AsUint(Dispatch::PREPASS) + (enableHitDistanceReconstruction ? 1 : 0);
        void* consts = PushDispatch(denoiserData, passIndex);
        AddSharedConstants_Relax(settings, consts);
    }

    { // TEMPORAL_ACCUMULATION
        uint32_t passIndex = AsUint(Dispatch::TEMPORAL_ACCUMULATION) + (m_CommonSettings.isDisocclusionThresholdMixAvailable ? 2 : 0) + (m_CommonSettings.isHistoryConfidenceAvailable ? 1 : 0);
        void* consts = PushDispatch(denoiserData, passIndex);
        AddSharedConstants_Relax(settings, consts);
    }

    { // HISTORY_FIX
        void* consts = PushDispatch(denoiserData, AsUint(Dispatch::HISTORY_FIX));
        AddSharedConstants_Relax(settings, consts);
    }

    { // HISTORY_CLAMPING
        void* consts = PushDispatch(denoiserData, AsUint(Dispatch::HISTORY_CLAMPING));
        AddSharedConstants_Relax(settings, consts);
    }

    if (settings.enableAntiFirefly) {
        { // COPY
            void* consts = PushDispatch(denoiserData, AsUint(Dispatch::COPY));
            AddSharedConstants_Relax(settings, consts);
        }

        { // ANTI_FIREFLY
            void* consts = PushDispatch(denoiserData, AsUint(Dispatch::ANTI_FIREFLY));
            AddSharedConstants_Relax(settings, consts);
        }
    }

    // A-TROUS
    for (uint32_t i = 0; i < iterationNum; i++) {
        uint32_t passIndex = AsUint(Dispatch::ATROUS) + (m_CommonSettings.isHistoryConfidenceAvailable ? RELAX_ATROUS_BINDING_VARIANT_NUM : 0);
        if (i != 0)
            passIndex += 2 - (i & 0x1);
        if (i == iterationNum - 1)
            passIndex += 2;

        RELAX_AtrousConstants* consts = (RELAX_AtrousConstants*)PushDispatch(denoiserData, AsUint(passIndex)); // TODO: same as "RELAX_AtrousSmemConstants"
        AddSharedConstants_Relax(settings, consts);
        consts->gStepSize = 1 << i;                          // TODO: push constant
        consts->gIsLastPass = i == iterationNum - 1 ? 1 : 0; // TODO: push constant
    }

    // SPLIT_SCREEN
    if (m_CommonSettings.splitScreen > 0.0f) {
        void* consts = PushDispatch(denoiserData, AsUint(Dispatch::SPLIT_SCREEN));
        AddSharedConstants_Relax(settings, consts);
    }

    // VALIDATION
    if (m_CommonSettings.enableValidation) {
        void* consts = PushDispatch(denoiserData, AsUint(Dispatch::VALIDATION));
        AddSharedConstants_Relax(settings, consts);
    }
}

// RELAX_SHARED
#ifdef NRD_EMBEDS_DXBC_SHADERS
#    include "RELAX_ClassifyTiles.cs.dxbc.h"
#    include "RELAX_Validation.cs.dxbc.h"
#endif

#ifdef NRD_EMBEDS_DXIL_SHADERS
#    include "RELAX_ClassifyTiles.cs.dxil.h"
#    include "RELAX_Validation.cs.dxil.h"
#endif

#ifdef NRD_EMBEDS_SPIRV_SHADERS
#    include "RELAX_ClassifyTiles.cs.spirv.h"
#    include "RELAX_Validation.cs.spirv.h"
#endif

// RELAX_DIFFUSE
#ifdef NRD_EMBEDS_DXBC_SHADERS
#    include "RELAX_Diffuse_AntiFirefly.cs.dxbc.h"
#    include "RELAX_Diffuse_Atrous.cs.dxbc.h"
#    include "RELAX_Diffuse_AtrousSmem.cs.dxbc.h"
#    include "RELAX_Diffuse_Copy.cs.dxbc.h"
#    include "RELAX_Diffuse_HistoryClamping.cs.dxbc.h"
#    include "RELAX_Diffuse_HistoryFix.cs.dxbc.h"
#    include "RELAX_Diffuse_HitDistReconstruction.cs.dxbc.h"
#    include "RELAX_Diffuse_HitDistReconstruction_5x5.cs.dxbc.h"
#    include "RELAX_Diffuse_PrePass.cs.dxbc.h"
#    include "RELAX_Diffuse_SplitScreen.cs.dxbc.h"
#    include "RELAX_Diffuse_TemporalAccumulation.cs.dxbc.h"
#endif

#ifdef NRD_EMBEDS_DXIL_SHADERS
#    include "RELAX_Diffuse_AntiFirefly.cs.dxil.h"
#    include "RELAX_Diffuse_Atrous.cs.dxil.h"
#    include "RELAX_Diffuse_AtrousSmem.cs.dxil.h"
#    include "RELAX_Diffuse_Copy.cs.dxil.h"
#    include "RELAX_Diffuse_HistoryClamping.cs.dxil.h"
#    include "RELAX_Diffuse_HistoryFix.cs.dxil.h"
#    include "RELAX_Diffuse_HitDistReconstruction.cs.dxil.h"
#    include "RELAX_Diffuse_HitDistReconstruction_5x5.cs.dxil.h"
#    include "RELAX_Diffuse_PrePass.cs.dxil.h"
#    include "RELAX_Diffuse_SplitScreen.cs.dxil.h"
#    include "RELAX_Diffuse_TemporalAccumulation.cs.dxil.h"
#endif

#ifdef NRD_EMBEDS_SPIRV_SHADERS
#    include "RELAX_Diffuse_AntiFirefly.cs.spirv.h"
#    include "RELAX_Diffuse_Atrous.cs.spirv.h"
#    include "RELAX_Diffuse_AtrousSmem.cs.spirv.h"
#    include "RELAX_Diffuse_Copy.cs.spirv.h"
#    include "RELAX_Diffuse_HistoryClamping.cs.spirv.h"
#    include "RELAX_Diffuse_HistoryFix.cs.spirv.h"
#    include "RELAX_Diffuse_HitDistReconstruction.cs.spirv.h"
#    include "RELAX_Diffuse_HitDistReconstruction_5x5.cs.spirv.h"
#    include "RELAX_Diffuse_PrePass.cs.spirv.h"
#    include "RELAX_Diffuse_SplitScreen.cs.spirv.h"
#    include "RELAX_Diffuse_TemporalAccumulation.cs.spirv.h"
#endif

#include "Denoisers/Relax_Diffuse.hpp"

// RELAX_DIFFUSE_SH
#ifdef NRD_EMBEDS_DXBC_SHADERS
#    include "RELAX_DiffuseSh_AntiFirefly.cs.dxbc.h"
#    include "RELAX_DiffuseSh_Atrous.cs.dxbc.h"
#    include "RELAX_DiffuseSh_AtrousSmem.cs.dxbc.h"
#    include "RELAX_DiffuseSh_Copy.cs.dxbc.h"
#    include "RELAX_DiffuseSh_HistoryClamping.cs.dxbc.h"
#    include "RELAX_DiffuseSh_HistoryFix.cs.dxbc.h"
#    include "RELAX_DiffuseSh_PrePass.cs.dxbc.h"
#    include "RELAX_DiffuseSh_SplitScreen.cs.dxbc.h"
#    include "RELAX_DiffuseSh_TemporalAccumulation.cs.dxbc.h"
#endif

#ifdef NRD_EMBEDS_DXIL_SHADERS
#    include "RELAX_DiffuseSh_AntiFirefly.cs.dxil.h"
#    include "RELAX_DiffuseSh_Atrous.cs.dxil.h"
#    include "RELAX_DiffuseSh_AtrousSmem.cs.dxil.h"
#    include "RELAX_DiffuseSh_Copy.cs.dxil.h"
#    include "RELAX_DiffuseSh_HistoryClamping.cs.dxil.h"
#    include "RELAX_DiffuseSh_HistoryFix.cs.dxil.h"
#    include "RELAX_DiffuseSh_PrePass.cs.dxil.h"
#    include "RELAX_DiffuseSh_SplitScreen.cs.dxil.h"
#    include "RELAX_DiffuseSh_TemporalAccumulation.cs.dxil.h"
#endif

#ifdef NRD_EMBEDS_SPIRV_SHADERS
#    include "RELAX_DiffuseSh_AntiFirefly.cs.spirv.h"
#    include "RELAX_DiffuseSh_Atrous.cs.spirv.h"
#    include "RELAX_DiffuseSh_AtrousSmem.cs.spirv.h"
#    include "RELAX_DiffuseSh_Copy.cs.spirv.h"
#    include "RELAX_DiffuseSh_HistoryClamping.cs.spirv.h"
#    include "RELAX_DiffuseSh_HistoryFix.cs.spirv.h"
#    include "RELAX_DiffuseSh_PrePass.cs.spirv.h"
#    include "RELAX_DiffuseSh_SplitScreen.cs.spirv.h"
#    include "RELAX_DiffuseSh_TemporalAccumulation.cs.spirv.h"
#endif

#include "Denoisers/Relax_DiffuseSh.hpp"

// RELAX_SPECULAR
#ifdef NRD_EMBEDS_DXBC_SHADERS
#    include "RELAX_Specular_AntiFirefly.cs.dxbc.h"
#    include "RELAX_Specular_Atrous.cs.dxbc.h"
#    include "RELAX_Specular_AtrousSmem.cs.dxbc.h"
#    include "RELAX_Specular_Copy.cs.dxbc.h"
#    include "RELAX_Specular_HistoryClamping.cs.dxbc.h"
#    include "RELAX_Specular_HistoryFix.cs.dxbc.h"
#    include "RELAX_Specular_HitDistReconstruction.cs.dxbc.h"
#    include "RELAX_Specular_HitDistReconstruction_5x5.cs.dxbc.h"
#    include "RELAX_Specular_PrePass.cs.dxbc.h"
#    include "RELAX_Specular_SplitScreen.cs.dxbc.h"
#    include "RELAX_Specular_TemporalAccumulation.cs.dxbc.h"
#endif

#ifdef NRD_EMBEDS_DXIL_SHADERS
#    include "RELAX_Specular_AntiFirefly.cs.dxil.h"
#    include "RELAX_Specular_Atrous.cs.dxil.h"
#    include "RELAX_Specular_AtrousSmem.cs.dxil.h"
#    include "RELAX_Specular_Copy.cs.dxil.h"
#    include "RELAX_Specular_HistoryClamping.cs.dxil.h"
#    include "RELAX_Specular_HistoryFix.cs.dxil.h"
#    include "RELAX_Specular_HitDistReconstruction.cs.dxil.h"
#    include "RELAX_Specular_HitDistReconstruction_5x5.cs.dxil.h"
#    include "RELAX_Specular_PrePass.cs.dxil.h"
#    include "RELAX_Specular_SplitScreen.cs.dxil.h"
#    include "RELAX_Specular_TemporalAccumulation.cs.dxil.h"
#endif

#ifdef NRD_EMBEDS_SPIRV_SHADERS
#    include "RELAX_Specular_AntiFirefly.cs.spirv.h"
#    include "RELAX_Specular_Atrous.cs.spirv.h"
#    include "RELAX_Specular_AtrousSmem.cs.spirv.h"
#    include "RELAX_Specular_Copy.cs.spirv.h"
#    include "RELAX_Specular_HistoryClamping.cs.spirv.h"
#    include "RELAX_Specular_HistoryFix.cs.spirv.h"
#    include "RELAX_Specular_HitDistReconstruction.cs.spirv.h"
#    include "RELAX_Specular_HitDistReconstruction_5x5.cs.spirv.h"
#    include "RELAX_Specular_PrePass.cs.spirv.h"
#    include "RELAX_Specular_SplitScreen.cs.spirv.h"
#    include "RELAX_Specular_TemporalAccumulation.cs.spirv.h"
#endif

#include "Denoisers/Relax_Specular.hpp"

// RELAX_SPECULAR_SH
#ifdef NRD_EMBEDS_DXBC_SHADERS
#    include "RELAX_SpecularSh_AntiFirefly.cs.dxbc.h"
#    include "RELAX_SpecularSh_Atrous.cs.dxbc.h"
#    include "RELAX_SpecularSh_AtrousSmem.cs.dxbc.h"
#    include "RELAX_SpecularSh_Copy.cs.dxbc.h"
#    include "RELAX_SpecularSh_HistoryClamping.cs.dxbc.h"
#    include "RELAX_SpecularSh_HistoryFix.cs.dxbc.h"
#    include "RELAX_SpecularSh_PrePass.cs.dxbc.h"
#    include "RELAX_SpecularSh_SplitScreen.cs.dxbc.h"
#    include "RELAX_SpecularSh_TemporalAccumulation.cs.dxbc.h"
#endif

#ifdef NRD_EMBEDS_DXIL_SHADERS
#    include "RELAX_SpecularSh_AntiFirefly.cs.dxil.h"
#    include "RELAX_SpecularSh_Atrous.cs.dxil.h"
#    include "RELAX_SpecularSh_AtrousSmem.cs.dxil.h"
#    include "RELAX_SpecularSh_Copy.cs.dxil.h"
#    include "RELAX_SpecularSh_HistoryClamping.cs.dxil.h"
#    include "RELAX_SpecularSh_HistoryFix.cs.dxil.h"
#    include "RELAX_SpecularSh_PrePass.cs.dxil.h"
#    include "RELAX_SpecularSh_SplitScreen.cs.dxil.h"
#    include "RELAX_SpecularSh_TemporalAccumulation.cs.dxil.h"
#endif

#ifdef NRD_EMBEDS_SPIRV_SHADERS
#    include "RELAX_SpecularSh_AntiFirefly.cs.spirv.h"
#    include "RELAX_SpecularSh_Atrous.cs.spirv.h"
#    include "RELAX_SpecularSh_AtrousSmem.cs.spirv.h"
#    include "RELAX_SpecularSh_Copy.cs.spirv.h"
#    include "RELAX_SpecularSh_HistoryClamping.cs.spirv.h"
#    include "RELAX_SpecularSh_HistoryFix.cs.spirv.h"
#    include "RELAX_SpecularSh_PrePass.cs.spirv.h"
#    include "RELAX_SpecularSh_SplitScreen.cs.spirv.h"
#    include "RELAX_SpecularSh_TemporalAccumulation.cs.spirv.h"
#endif

#include "Denoisers/Relax_SpecularSh.hpp"

// RELAX_DIFFUSE_SPECULAR
#ifdef NRD_EMBEDS_DXBC_SHADERS
#    include "RELAX_DiffuseSpecular_AntiFirefly.cs.dxbc.h"
#    include "RELAX_DiffuseSpecular_Atrous.cs.dxbc.h"
#    include "RELAX_DiffuseSpecular_AtrousSmem.cs.dxbc.h"
#    include "RELAX_DiffuseSpecular_Copy.cs.dxbc.h"
#    include "RELAX_DiffuseSpecular_HistoryClamping.cs.dxbc.h"
#    include "RELAX_DiffuseSpecular_HistoryFix.cs.dxbc.h"
#    include "RELAX_DiffuseSpecular_HitDistReconstruction.cs.dxbc.h"
#    include "RELAX_DiffuseSpecular_HitDistReconstruction_5x5.cs.dxbc.h"
#    include "RELAX_DiffuseSpecular_PrePass.cs.dxbc.h"
#    include "RELAX_DiffuseSpecular_SplitScreen.cs.dxbc.h"
#    include "RELAX_DiffuseSpecular_TemporalAccumulation.cs.dxbc.h"
#endif

#ifdef NRD_EMBEDS_DXIL_SHADERS
#    include "RELAX_DiffuseSpecular_AntiFirefly.cs.dxil.h"
#    include "RELAX_DiffuseSpecular_Atrous.cs.dxil.h"
#    include "RELAX_DiffuseSpecular_AtrousSmem.cs.dxil.h"
#    include "RELAX_DiffuseSpecular_Copy.cs.dxil.h"
#    include "RELAX_DiffuseSpecular_HistoryClamping.cs.dxil.h"
#    include "RELAX_DiffuseSpecular_HistoryFix.cs.dxil.h"
#    include "RELAX_DiffuseSpecular_HitDistReconstruction.cs.dxil.h"
#    include "RELAX_DiffuseSpecular_HitDistReconstruction_5x5.cs.dxil.h"
#    include "RELAX_DiffuseSpecular_PrePass.cs.dxil.h"
#    include "RELAX_DiffuseSpecular_SplitScreen.cs.dxil.h"
#    include "RELAX_DiffuseSpecular_TemporalAccumulation.cs.dxil.h"
#endif

#ifdef NRD_EMBEDS_SPIRV_SHADERS
#    include "RELAX_DiffuseSpecular_AntiFirefly.cs.spirv.h"
#    include "RELAX_DiffuseSpecular_Atrous.cs.spirv.h"
#    include "RELAX_DiffuseSpecular_AtrousSmem.cs.spirv.h"
#    include "RELAX_DiffuseSpecular_Copy.cs.spirv.h"
#    include "RELAX_DiffuseSpecular_HistoryClamping.cs.spirv.h"
#    include "RELAX_DiffuseSpecular_HistoryFix.cs.spirv.h"
#    include "RELAX_DiffuseSpecular_HitDistReconstruction.cs.spirv.h"
#    include "RELAX_DiffuseSpecular_HitDistReconstruction_5x5.cs.spirv.h"
#    include "RELAX_DiffuseSpecular_PrePass.cs.spirv.h"
#    include "RELAX_DiffuseSpecular_SplitScreen.cs.spirv.h"
#    include "RELAX_DiffuseSpecular_TemporalAccumulation.cs.spirv.h"
#endif

#include "Denoisers/Relax_DiffuseSpecular.hpp"

// RELAX_DIFFUSE_SPECULAR_SH
#ifdef NRD_EMBEDS_DXBC_SHADERS
#    include "RELAX_DiffuseSpecularSh_AntiFirefly.cs.dxbc.h"
#    include "RELAX_DiffuseSpecularSh_Atrous.cs.dxbc.h"
#    include "RELAX_DiffuseSpecularSh_AtrousSmem.cs.dxbc.h"
#    include "RELAX_DiffuseSpecularSh_Copy.cs.dxbc.h"
#    include "RELAX_DiffuseSpecularSh_HistoryClamping.cs.dxbc.h"
#    include "RELAX_DiffuseSpecularSh_HistoryFix.cs.dxbc.h"
#    include "RELAX_DiffuseSpecularSh_PrePass.cs.dxbc.h"
#    include "RELAX_DiffuseSpecularSh_SplitScreen.cs.dxbc.h"
#    include "RELAX_DiffuseSpecularSh_TemporalAccumulation.cs.dxbc.h"
#endif

#ifdef NRD_EMBEDS_DXIL_SHADERS
#    include "RELAX_DiffuseSpecularSh_AntiFirefly.cs.dxil.h"
#    include "RELAX_DiffuseSpecularSh_Atrous.cs.dxil.h"
#    include "RELAX_DiffuseSpecularSh_AtrousSmem.cs.dxil.h"
#    include "RELAX_DiffuseSpecularSh_Copy.cs.dxil.h"
#    include "RELAX_DiffuseSpecularSh_HistoryClamping.cs.dxil.h"
#    include "RELAX_DiffuseSpecularSh_HistoryFix.cs.dxil.h"
#    include "RELAX_DiffuseSpecularSh_PrePass.cs.dxil.h"
#    include "RELAX_DiffuseSpecularSh_SplitScreen.cs.dxil.h"
#    include "RELAX_DiffuseSpecularSh_TemporalAccumulation.cs.dxil.h"
#endif

#ifdef NRD_EMBEDS_SPIRV_SHADERS
#    include "RELAX_DiffuseSpecularSh_AntiFirefly.cs.spirv.h"
#    include "RELAX_DiffuseSpecularSh_Atrous.cs.spirv.h"
#    include "RELAX_DiffuseSpecularSh_AtrousSmem.cs.spirv.h"
#    include "RELAX_DiffuseSpecularSh_Copy.cs.spirv.h"
#    include "RELAX_DiffuseSpecularSh_HistoryClamping.cs.spirv.h"
#    include "RELAX_DiffuseSpecularSh_HistoryFix.cs.spirv.h"
#    include "RELAX_DiffuseSpecularSh_PrePass.cs.spirv.h"
#    include "RELAX_DiffuseSpecularSh_SplitScreen.cs.spirv.h"
#    include "RELAX_DiffuseSpecularSh_TemporalAccumulation.cs.spirv.h"
#endif

#include "Denoisers/Relax_DiffuseSpecularSh.hpp"
