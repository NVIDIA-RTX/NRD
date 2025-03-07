/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "InstanceImpl.h"

#include <array>

#include "../Shaders/Include/REBLUR_Config.hlsli"
#include "../Shaders/Resources/REBLUR_Blur.resources.hlsli"
#include "../Shaders/Resources/REBLUR_ClassifyTiles.resources.hlsli"
#include "../Shaders/Resources/REBLUR_HistoryFix.resources.hlsli"
#include "../Shaders/Resources/REBLUR_HitDistReconstruction.resources.hlsli"
#include "../Shaders/Resources/REBLUR_PostBlur.resources.hlsli"
#include "../Shaders/Resources/REBLUR_PrePass.resources.hlsli"
#include "../Shaders/Resources/REBLUR_SplitScreen.resources.hlsli"
#include "../Shaders/Resources/REBLUR_TemporalAccumulation.resources.hlsli"
#include "../Shaders/Resources/REBLUR_TemporalStabilization.resources.hlsli"
#include "../Shaders/Resources/REBLUR_Validation.resources.hlsli"

// Permutations
#define REBLUR_HITDIST_RECONSTRUCTION_PERMUTATION_NUM               4
#define REBLUR_PREPASS_PERMUTATION_NUM                              2
#define REBLUR_TEMPORAL_ACCUMULATION_PERMUTATION_NUM                8
#define REBLUR_POST_BLUR_PERMUTATION_NUM                            2
#define REBLUR_TEMPORAL_STABILIZATION_PERMUTATION_NUM               2

#define REBLUR_OCCLUSION_HITDIST_RECONSTRUCTION_PERMUTATION_NUM     2
#define REBLUR_OCCLUSION_TEMPORAL_ACCUMULATION_PERMUTATION_NUM      8

// Formats
#define REBLUR_FORMAT                                               Format::RGBA16_SFLOAT // .xyz - color, .w - normalized hit distance
#define REBLUR_FORMAT_FAST_HISTORY                                  Format::R16_SFLOAT // .x - luminance

// TODO: there can be an option for UNORM/SNORM 8/16-bit promotion/demotion, but only for resources marked as main/fast history (i.e. not normal-roughness and internal data resources)
#define REBLUR_FORMAT_OCCLUSION                                     Format::R16_UNORM
#define REBLUR_FORMAT_OCCLUSION_FAST_HISTORY                        Format::R8_UNORM // TODO: keep an eye on precision, but can be even used for the main history if accumulation is not as long

#define REBLUR_FORMAT_DIRECTIONAL_OCCLUSION                         Format::RGBA16_SNORM
#define REBLUR_FORMAT_DIRECTIONAL_OCCLUSION_FAST_HISTORY            REBLUR_FORMAT_OCCLUSION_FAST_HISTORY

#define REBLUR_FORMAT_PREV_VIEWZ                                    Format::R32_SFLOAT
#define REBLUR_FORMAT_PREV_INTERNAL_DATA                            Format::R16_UINT

#define REBLUR_FORMAT_TILES                                         Format::R8_UNORM

#if (NRD_NORMAL_ENCODING == 0)
    #define REBLUR_FORMAT_PREV_NORMAL_ROUGHNESS                     Format::RGBA8_UNORM
#elif (NRD_NORMAL_ENCODING == 1)
    #define REBLUR_FORMAT_PREV_NORMAL_ROUGHNESS                     Format::RGBA8_SNORM
#elif (NRD_NORMAL_ENCODING == 2)
    #define REBLUR_FORMAT_PREV_NORMAL_ROUGHNESS                     Format::R10_G10_B10_A2_UNORM
#elif (NRD_NORMAL_ENCODING == 3)
    #define REBLUR_FORMAT_PREV_NORMAL_ROUGHNESS                     Format::RGBA16_UNORM
#elif (NRD_NORMAL_ENCODING == 4)
    #define REBLUR_FORMAT_PREV_NORMAL_ROUGHNESS                     Format::RGBA16_SFLOAT
#endif

#define REBLUR_FORMAT_HITDIST_FOR_TRACKING                          Format::R16_SFLOAT

// Other
#define REBLUR_DUMMY                                                AsUint(ResourceType::IN_VIEWZ)
#define REBLUR_NO_PERMUTATIONS                                      1

#define REBLUR_ADD_VALIDATION_DISPATCH( data2, diff, spec ) \
    PushPass("Validation"); \
    { \
        PushInput( AsUint(ResourceType::IN_NORMAL_ROUGHNESS) ); \
        PushInput( AsUint(ResourceType::IN_VIEWZ) ); \
        PushInput( AsUint(ResourceType::IN_MV) ); \
        PushInput( AsUint(Transient::DATA1) ); \
        PushInput( AsUint(data2) ); \
        PushInput( AsUint(diff) ); \
        PushInput( AsUint(spec) ); \
        PushOutput( AsUint(ResourceType::OUT_VALIDATION) ); \
        AddDispatch( REBLUR_Validation, REBLUR_Validation, IGNORE_RS ); \
    }

struct ReblurProps
{
    bool hasDiffuse;
    bool hasSpecular;
};

constexpr std::array<ReblurProps, 10> g_ReblurProps =
{{
    {true, false},      // REBLUR_DIFFUSE
    {true, false},      // REBLUR_DIFFUSE_OCCLUSION
    {true, false},      // REBLUR_DIFFUSE_SH
    {false, true},      // REBLUR_SPECULAR
    {false, true},      // REBLUR_SPECULAR_OCCLUSION
    {false, true},      // REBLUR_SPECULAR_SH
    {true, true},       // REBLUR_DIFFUSE_SPECULAR
    {true, true},       // REBLUR_DIFFUSE_SPECULAR_OCCLUSION
    {true, true},       // REBLUR_DIFFUSE_SPECULAR_SH
    {true, false},      // REBLUR_DIFFUSE_DIRECTIONAL_OCCLUSION
}};

void nrd::InstanceImpl::Update_Reblur(const DenoiserData& denoiserData)
{
    enum class Dispatch
    {
        CLASSIFY_TILES,
        HITDIST_RECONSTRUCTION  = CLASSIFY_TILES + REBLUR_NO_PERMUTATIONS * 1, // CLASSIFY_TILES doesn't have perf mode
        PREPASS                 = HITDIST_RECONSTRUCTION + REBLUR_HITDIST_RECONSTRUCTION_PERMUTATION_NUM * 2,
        TEMPORAL_ACCUMULATION   = PREPASS + REBLUR_PREPASS_PERMUTATION_NUM * 2,
        HISTORY_FIX             = TEMPORAL_ACCUMULATION + REBLUR_TEMPORAL_ACCUMULATION_PERMUTATION_NUM * 2,
        BLUR                    = HISTORY_FIX + REBLUR_NO_PERMUTATIONS * 2,
        POST_BLUR               = BLUR + REBLUR_NO_PERMUTATIONS * 2,
        TEMPORAL_STABILIZATION  = POST_BLUR + REBLUR_POST_BLUR_PERMUTATION_NUM * 2,
        SPLIT_SCREEN            = TEMPORAL_STABILIZATION + REBLUR_TEMPORAL_STABILIZATION_PERMUTATION_NUM * 2,
        VALIDATION              = SPLIT_SCREEN + REBLUR_NO_PERMUTATIONS * 1, // SPLIT_SCREEN doesn't have perf mode
    };

    NRD_DECLARE_DIMS;

    const ReblurSettings& settings = denoiserData.settings.reblur;
    const ReblurProps& props = g_ReblurProps[ size_t(denoiserData.desc.denoiser) - size_t(Denoiser::REBLUR_DIFFUSE) ];

    bool enableHitDistanceReconstruction = settings.hitDistanceReconstructionMode != HitDistanceReconstructionMode::OFF && settings.checkerboardMode == CheckerboardMode::OFF;
    bool skipTemporalStabilization = settings.maxStabilizedFrameNum == 0;
    bool skipPrePass = (settings.diffusePrepassBlurRadius == 0.0f || !props.hasDiffuse) &&
        (settings.specularPrepassBlurRadius == 0.0f || !props.hasSpecular) &&
        settings.checkerboardMode == CheckerboardMode::OFF;

    // SPLIT_SCREEN (passthrough)
    if (m_CommonSettings.splitScreen >= 1.0f)
    {
        void* consts = PushDispatch(denoiserData, AsUint(Dispatch::SPLIT_SCREEN));
        AddSharedConstants_Reblur(settings, consts);

        return;
    }

    { // CLASSIFY_TILES
        void* consts = PushDispatch(denoiserData, AsUint(Dispatch::CLASSIFY_TILES));
        AddSharedConstants_Reblur(settings, consts);
    }

    // HITDIST_RECONSTRUCTION
    if (enableHitDistanceReconstruction)
    {
        uint32_t passIndex = AsUint(Dispatch::HITDIST_RECONSTRUCTION) + (settings.hitDistanceReconstructionMode == HitDistanceReconstructionMode::AREA_5X5 ? 4 : 0) + (!skipPrePass ? 2 : 0) + (settings.enablePerformanceMode ? 1 : 0);
        void* consts = PushDispatch(denoiserData, passIndex);
        AddSharedConstants_Reblur(settings, consts);
    }

    // PREPASS
    if (!skipPrePass)
    {
        uint32_t passIndex = AsUint(Dispatch::PREPASS) + (enableHitDistanceReconstruction ? 2 : 0) + (settings.enablePerformanceMode ? 1 : 0);
        void* consts = PushDispatch(denoiserData, passIndex);
        AddSharedConstants_Reblur(settings, consts);
    }

    { // TEMPORAL_ACCUMULATION
        uint32_t passIndex = AsUint(Dispatch::TEMPORAL_ACCUMULATION) + (m_CommonSettings.isDisocclusionThresholdMixAvailable ? 8 : 0) +
            (m_CommonSettings.isHistoryConfidenceAvailable ? 4 : 0) +
            ((!skipPrePass || enableHitDistanceReconstruction) ? 2 : 0) + (settings.enablePerformanceMode ? 1 : 0);
        void* consts = PushDispatch(denoiserData, passIndex);
        AddSharedConstants_Reblur(settings, consts);
    }

    { // HISTORY_FIX
        uint32_t passIndex = AsUint(Dispatch::HISTORY_FIX) + (settings.enablePerformanceMode ? 1 : 0);
        void* consts = PushDispatch(denoiserData, passIndex);
        AddSharedConstants_Reblur(settings, consts);
    }

    { // BLUR
        uint32_t passIndex = AsUint(Dispatch::BLUR) + (settings.enablePerformanceMode ? 1 : 0);
        void* consts = PushDispatch(denoiserData, passIndex);
        AddSharedConstants_Reblur(settings, consts);
    }

    { // POST_BLUR
        uint32_t passIndex = AsUint(Dispatch::POST_BLUR) + (skipTemporalStabilization ? 0 : 2) + (settings.enablePerformanceMode ? 1 : 0);
        void* consts = PushDispatch(denoiserData, passIndex);
        AddSharedConstants_Reblur(settings, consts);
    }

    // TEMPORAL_STABILIZATION
    if (!skipTemporalStabilization)
    {
        uint32_t passIndex = AsUint(Dispatch::TEMPORAL_STABILIZATION) + (m_CommonSettings.isBaseColorMetalnessAvailable ? 2 : 0) + (settings.enablePerformanceMode ? 1 : 0);
        void* consts = PushDispatch(denoiserData, passIndex);
        AddSharedConstants_Reblur(settings, consts);
    }

    // SPLIT_SCREEN
    if (m_CommonSettings.splitScreen > 0.0f)
    {
        void* consts = PushDispatch(denoiserData, AsUint(Dispatch::SPLIT_SCREEN));
        AddSharedConstants_Reblur(settings, consts);
    }

    // VALIDATION
    if (m_CommonSettings.enableValidation)
    {
        REBLUR_ValidationConstants* consts = (REBLUR_ValidationConstants*)PushDispatch(denoiserData, AsUint(Dispatch::VALIDATION));
        AddSharedConstants_Reblur(settings, consts);
        consts->gHasDiffuse = props.hasDiffuse ? 1 : 0; // TODO: push constant
        consts->gHasSpecular = props.hasSpecular ? 1 : 0; // TODO: push constant
    }
}

void nrd::InstanceImpl::Update_ReblurOcclusion(const DenoiserData& denoiserData)
{
    enum class Dispatch
    {
        CLASSIFY_TILES,
        HITDIST_RECONSTRUCTION  = CLASSIFY_TILES + REBLUR_NO_PERMUTATIONS * 1, // CLASSIFY_TILES doesn't have perf mode
        TEMPORAL_ACCUMULATION   = HITDIST_RECONSTRUCTION + REBLUR_OCCLUSION_HITDIST_RECONSTRUCTION_PERMUTATION_NUM * 2,
        HISTORY_FIX             = TEMPORAL_ACCUMULATION + REBLUR_OCCLUSION_TEMPORAL_ACCUMULATION_PERMUTATION_NUM * 2,
        BLUR                    = HISTORY_FIX + REBLUR_NO_PERMUTATIONS * 2,
        POST_BLUR               = BLUR + REBLUR_NO_PERMUTATIONS * 2,
        SPLIT_SCREEN            = POST_BLUR + REBLUR_NO_PERMUTATIONS * 2,
        VALIDATION              = SPLIT_SCREEN + REBLUR_NO_PERMUTATIONS * 1, // SPLIT_SCREEN doesn't have perf mode
    };

    NRD_DECLARE_DIMS;

    const ReblurSettings& settings = denoiserData.settings.reblur;
    const ReblurProps& props = g_ReblurProps[ size_t(denoiserData.desc.denoiser) - size_t(Denoiser::REBLUR_DIFFUSE) ];

    bool enableHitDistanceReconstruction = settings.hitDistanceReconstructionMode != HitDistanceReconstructionMode::OFF && settings.checkerboardMode == CheckerboardMode::OFF;

    // SPLIT_SCREEN (passthrough)
    if (m_CommonSettings.splitScreen >= 1.0f)
    {
        void* consts = PushDispatch(denoiserData, AsUint(Dispatch::SPLIT_SCREEN));
        AddSharedConstants_Reblur(settings, consts);

        return;
    }

    { // CLASSIFY_TILES
        void* consts = PushDispatch(denoiserData, AsUint(Dispatch::CLASSIFY_TILES));
        AddSharedConstants_Reblur(settings, consts);
    }

    // HITDIST_RECONSTRUCTION
    if (enableHitDistanceReconstruction)
    {
        uint32_t passIndex = AsUint(Dispatch::HITDIST_RECONSTRUCTION) + (settings.hitDistanceReconstructionMode == HitDistanceReconstructionMode::AREA_5X5 ? 2 : 0) + (settings.enablePerformanceMode ? 1 : 0);
        void* consts = PushDispatch(denoiserData, passIndex);
        AddSharedConstants_Reblur(settings, consts);
    }

    { // TEMPORAL_ACCUMULATION
        uint32_t passIndex = AsUint(Dispatch::TEMPORAL_ACCUMULATION) + (m_CommonSettings.isDisocclusionThresholdMixAvailable ? 8 : 0) +
            (m_CommonSettings.isHistoryConfidenceAvailable ? 4 : 0) + (enableHitDistanceReconstruction ? 2 : 0) + (settings.enablePerformanceMode ? 1 : 0);
        void* consts = PushDispatch(denoiserData, passIndex);
        AddSharedConstants_Reblur(settings, consts);
    }

    { // HISTORY_FIX
        uint32_t passIndex = AsUint(Dispatch::HISTORY_FIX) + (!settings.enableAntiFirefly ? 1 : 0);
        void* consts = PushDispatch(denoiserData, passIndex);
        AddSharedConstants_Reblur(settings, consts);
    }

    { // BLUR
        uint32_t passIndex = AsUint(Dispatch::BLUR) + (settings.enablePerformanceMode ? 1 : 0);
        void* consts = PushDispatch(denoiserData, passIndex);
        AddSharedConstants_Reblur(settings, consts);
    }

    { // POST_BLUR
        uint32_t passIndex = AsUint(Dispatch::POST_BLUR) + (settings.enablePerformanceMode ? 1 : 0);
        void* consts = PushDispatch(denoiserData, passIndex);
        AddSharedConstants_Reblur(settings, consts);
    }

    // SPLIT_SCREEN
    if (m_CommonSettings.splitScreen > 0.0f)
    {
        void* consts = PushDispatch(denoiserData, AsUint(Dispatch::SPLIT_SCREEN));
        AddSharedConstants_Reblur(settings, consts);
    }

    // VALIDATION
    if (m_CommonSettings.enableValidation)
    {
        REBLUR_ValidationConstants* consts = (REBLUR_ValidationConstants*)PushDispatch(denoiserData, AsUint(Dispatch::VALIDATION));
        AddSharedConstants_Reblur(settings, consts);
        consts->gHasDiffuse = props.hasDiffuse ? 1 : 0; // TODO: push constant
        consts->gHasSpecular = props.hasSpecular ? 1 : 0; // TODO: push constant
    }
}

void nrd::InstanceImpl::AddSharedConstants_Reblur(const ReblurSettings& settings, void* data)
{
    struct SharedConstants
    {
        REBLUR_SHARED_CONSTANTS
    };

    NRD_DECLARE_DIMS;

    bool isRectChanged = rectW != rectWprev || rectH != rectHprev;
    bool isHistoryReset = m_CommonSettings.accumulationMode != AccumulationMode::CONTINUE;
    float unproject = 1.0f / (0.5f * rectH * m_ProjectY);
    float worstResolutionScale = min(float(rectW) / float(resourceW), float(rectH) / float(resourceH));
    float maxBlurRadius = settings.maxBlurRadius * worstResolutionScale;
    float diffusePrepassBlurRadius = settings.diffusePrepassBlurRadius * worstResolutionScale;
    float specularPrepassBlurRadius = settings.specularPrepassBlurRadius * worstResolutionScale;
    float disocclusionThresholdBonus = (1.0f + m_JitterDelta) / float(rectH);
    float stabilizationStrength = settings.maxStabilizedFrameNum / (1.0f + settings.maxStabilizedFrameNum);
    uint32_t maxAccumulatedFrameNum = min(settings.maxAccumulatedFrameNum, REBLUR_MAX_HISTORY_FRAME_NUM);

    uint32_t diffCheckerboard = 2;
    uint32_t specCheckerboard = 2;
    switch (settings.checkerboardMode)
    {
        case CheckerboardMode::BLACK:
            diffCheckerboard = 0;
            specCheckerboard = 1;
            break;
        case CheckerboardMode::WHITE:
            diffCheckerboard = 1;
            specCheckerboard = 0;
            break;
    }

    SharedConstants* consts                                     = (SharedConstants*)data;
    consts->gWorldToClip                                        = m_WorldToClip;
    consts->gViewToClip                                         = m_ViewToClip;
    consts->gViewToWorld                                        = m_ViewToWorld;
    consts->gWorldToViewPrev                                    = m_WorldToViewPrev;
    consts->gWorldToClipPrev                                    = m_WorldToClipPrev;
    consts->gWorldPrevToWorld                                   = m_WorldPrevToWorld;
    consts->gRotatorPre                                         = m_RotatorPre;
    consts->gRotator                                            = m_Rotator;
    consts->gRotatorPost                                        = m_RotatorPost;
    consts->gFrustum                                            = m_Frustum;
    consts->gFrustumPrev                                        = m_FrustumPrev;
    consts->gCameraDelta                                        = m_CameraDelta.xmm;
    consts->gHitDistParams                                      = float4(settings.hitDistanceParameters.A, settings.hitDistanceParameters.B, settings.hitDistanceParameters.C, settings.hitDistanceParameters.D);
    consts->gViewVectorWorld                                    = m_ViewDirection.xmm;
    consts->gViewVectorWorldPrev                                = m_ViewDirectionPrev.xmm;
    consts->gMvScale                                            = float4(m_CommonSettings.motionVectorScale[0], m_CommonSettings.motionVectorScale[1], m_CommonSettings.motionVectorScale[2], m_CommonSettings.isMotionVectorInWorldSpace ? 1.0f : 0.0f);
    consts->gAntilagParams                                      = float2(settings.antilagSettings.luminanceSigmaScale, settings.antilagSettings.luminanceSensitivity);
    consts->gResourceSize                                       = float2(float(resourceW), float(resourceH));
    consts->gResourceSizeInv                                    = float2(1.0f / float(resourceW), 1.0f / float(resourceH));
    consts->gResourceSizeInvPrev                                = float2(1.0f / float(resourceWprev), 1.0f / float(resourceHprev));
    consts->gRectSize                                           = float2(float(rectW), float(rectH));
    consts->gRectSizeInv                                        = float2(1.0f / float(rectW), 1.0f / float(rectH));
    consts->gRectSizePrev                                       = float2(float(rectWprev), float(rectHprev));
    consts->gResolutionScale                                    = float2(float(rectW) / float(resourceW), float(rectH) / float(resourceH));
    consts->gResolutionScalePrev                                = float2(float(rectWprev) / float(resourceWprev), float(rectHprev) / float(resourceHprev));
    consts->gRectOffset                                         = float2(float(m_CommonSettings.rectOrigin[0]) / float(resourceW), float(m_CommonSettings.rectOrigin[1]) / float(resourceH));
    consts->gSpecProbabilityThresholdsForMvModification         = float2(m_CommonSettings.isBaseColorMetalnessAvailable ? settings.specularProbabilityThresholdsForMvModification[0] : 2.0f, m_CommonSettings.isBaseColorMetalnessAvailable ? settings.specularProbabilityThresholdsForMvModification[1] : 3.0f);
    consts->gJitter                                             = float2(m_CommonSettings.cameraJitter[0], m_CommonSettings.cameraJitter[1]);
    consts->gPrintfAt                                           = uint2(m_CommonSettings.printfAt[0], m_CommonSettings.printfAt[1]);
    consts->gRectOrigin                                         = uint2(m_CommonSettings.rectOrigin[0], m_CommonSettings.rectOrigin[1]);
    consts->gRectSizeMinusOne                                   = int2(rectW - 1, rectH - 1);
    consts->gDisocclusionThreshold                              = m_CommonSettings.disocclusionThreshold + disocclusionThresholdBonus;
    consts->gDisocclusionThresholdAlternate                     = m_CommonSettings.disocclusionThresholdAlternate + disocclusionThresholdBonus;
    consts->gCameraAttachedReflectionMaterialID                 = m_CommonSettings.cameraAttachedReflectionMaterialID;
    consts->gStrandMaterialID                                   = m_CommonSettings.strandMaterialID;
    consts->gStrandThickness                                    = m_CommonSettings.strandThickness;
    consts->gStabilizationStrength                              = isHistoryReset ? 0.0f : stabilizationStrength;
    consts->gDebug                                              = m_CommonSettings.debug;
    consts->gOrthoMode                                          = m_OrthoMode;
    consts->gUnproject                                          = unproject;
    consts->gDenoisingRange                                     = m_CommonSettings.denoisingRange;
    consts->gPlaneDistSensitivity                               = settings.planeDistanceSensitivity;
    consts->gFramerateScale                                     = m_FrameRateScale;
    consts->gMaxBlurRadius                                      = max(maxBlurRadius, settings.minBlurRadius);
    consts->gMinBlurRadius                                      = settings.minBlurRadius;
    consts->gDiffPrepassBlurRadius                              = diffusePrepassBlurRadius;
    consts->gSpecPrepassBlurRadius                              = specularPrepassBlurRadius;
    consts->gMaxAccumulatedFrameNum                             = isHistoryReset ? 0 : float(maxAccumulatedFrameNum);
    consts->gMaxFastAccumulatedFrameNum                         = isHistoryReset ? 0 : float(settings.maxFastAccumulatedFrameNum);
    consts->gAntiFirefly                                        = settings.enableAntiFirefly ? 1.0f : 0.0f;
    consts->gLobeAngleFraction                                  = settings.lobeAngleFraction * settings.lobeAngleFraction; // TODO: GetSpecularLobeTanHalfAngle has been fixed, but we want to use existing settings
    consts->gRoughnessFraction                                  = settings.roughnessFraction;
    consts->gResponsiveAccumulationRoughnessThreshold           = settings.responsiveAccumulationRoughnessThreshold;
    consts->gHistoryFixFrameNum                                 = (float)settings.historyFixFrameNum;
    consts->gHistoryFixBasePixelStride                          = (float)settings.historyFixBasePixelStride;
    consts->gMinRectDimMulUnproject                             = (float)min(rectW, rectH) * unproject;
    consts->gUsePrepassNotOnlyForSpecularMotionEstimation       = settings.usePrepassOnlyForSpecularMotionEstimation ? 0.0f : 1.0f;
    consts->gSplitScreen                                        = m_CommonSettings.splitScreen;
    consts->gSplitScreenPrev                                    = m_SplitScreenPrev;
    consts->gCheckerboardResolveAccumSpeed                      = m_CheckerboardResolveAccumSpeed;
    consts->gViewZScale                                         = m_CommonSettings.viewZScale;
    consts->gFireflySuppressorMinRelativeScale                  = settings.fireflySuppressorMinRelativeScale;
    consts->gMinHitDistanceWeight                               = settings.minHitDistanceWeight;
    consts->gDiffMinMaterial                                    = settings.minMaterialForDiffuse;
    consts->gSpecMinMaterial                                    = settings.minMaterialForSpecular;
    consts->gHasHistoryConfidence                               = m_CommonSettings.isHistoryConfidenceAvailable;
    consts->gHasDisocclusionThresholdMix                        = m_CommonSettings.isDisocclusionThresholdMixAvailable;
    consts->gDiffCheckerboard                                   = diffCheckerboard;
    consts->gSpecCheckerboard                                   = specCheckerboard;
    consts->gFrameIndex                                         = m_CommonSettings.frameIndex;
    consts->gIsRectChanged                                      = isRectChanged ? 1 : 0;
    consts->gResetHistory                                       = isHistoryReset ? 1 : 0;
}

// REBLUR_SHARED
#ifdef NRD_EMBEDS_DXBC_SHADERS
    #include "REBLUR_ClassifyTiles.cs.dxbc.h"
    #include "REBLUR_Validation.cs.dxbc.h"
#endif

#ifdef NRD_EMBEDS_DXIL_SHADERS
    #include "REBLUR_ClassifyTiles.cs.dxil.h"
    #include "REBLUR_Validation.cs.dxil.h"
#endif

#ifdef NRD_EMBEDS_SPIRV_SHADERS
    #include "REBLUR_ClassifyTiles.cs.spirv.h"
    #include "REBLUR_Validation.cs.spirv.h"
#endif

// REBLUR_DIFFUSE
#ifdef NRD_EMBEDS_DXBC_SHADERS
    #include "REBLUR_Diffuse_HitDistReconstruction.cs.dxbc.h"
    #include "REBLUR_Diffuse_HitDistReconstruction_5x5.cs.dxbc.h"
    #include "REBLUR_Diffuse_PrePass.cs.dxbc.h"
    #include "REBLUR_Diffuse_TemporalAccumulation.cs.dxbc.h"
    #include "REBLUR_Diffuse_HistoryFix.cs.dxbc.h"
    #include "REBLUR_Diffuse_Blur.cs.dxbc.h"
    #include "REBLUR_Diffuse_PostBlur.cs.dxbc.h"
    #include "REBLUR_Diffuse_PostBlur_NoTemporalStabilization.cs.dxbc.h"
    #include "REBLUR_Diffuse_TemporalStabilization.cs.dxbc.h"
    #include "REBLUR_Diffuse_SplitScreen.cs.dxbc.h"

    #include "REBLUR_Perf_Diffuse_HitDistReconstruction.cs.dxbc.h"
    #include "REBLUR_Perf_Diffuse_HitDistReconstruction_5x5.cs.dxbc.h"
    #include "REBLUR_Perf_Diffuse_PrePass.cs.dxbc.h"
    #include "REBLUR_Perf_Diffuse_TemporalAccumulation.cs.dxbc.h"
    #include "REBLUR_Perf_Diffuse_HistoryFix.cs.dxbc.h"
    #include "REBLUR_Perf_Diffuse_Blur.cs.dxbc.h"
    #include "REBLUR_Perf_Diffuse_PostBlur.cs.dxbc.h"
    #include "REBLUR_Perf_Diffuse_PostBlur_NoTemporalStabilization.cs.dxbc.h"
    #include "REBLUR_Perf_Diffuse_TemporalStabilization.cs.dxbc.h"
#endif

#ifdef NRD_EMBEDS_DXIL_SHADERS
    #include "REBLUR_Diffuse_HitDistReconstruction.cs.dxil.h"
    #include "REBLUR_Diffuse_HitDistReconstruction_5x5.cs.dxil.h"
    #include "REBLUR_Diffuse_PrePass.cs.dxil.h"
    #include "REBLUR_Diffuse_TemporalAccumulation.cs.dxil.h"
    #include "REBLUR_Diffuse_HistoryFix.cs.dxil.h"
    #include "REBLUR_Diffuse_Blur.cs.dxil.h"
    #include "REBLUR_Diffuse_PostBlur.cs.dxil.h"
    #include "REBLUR_Diffuse_PostBlur_NoTemporalStabilization.cs.dxil.h"
    #include "REBLUR_Diffuse_TemporalStabilization.cs.dxil.h"
    #include "REBLUR_Diffuse_SplitScreen.cs.dxil.h"

    #include "REBLUR_Perf_Diffuse_HitDistReconstruction.cs.dxil.h"
    #include "REBLUR_Perf_Diffuse_HitDistReconstruction_5x5.cs.dxil.h"
    #include "REBLUR_Perf_Diffuse_PrePass.cs.dxil.h"
    #include "REBLUR_Perf_Diffuse_TemporalAccumulation.cs.dxil.h"
    #include "REBLUR_Perf_Diffuse_HistoryFix.cs.dxil.h"
    #include "REBLUR_Perf_Diffuse_Blur.cs.dxil.h"
    #include "REBLUR_Perf_Diffuse_PostBlur.cs.dxil.h"
    #include "REBLUR_Perf_Diffuse_PostBlur_NoTemporalStabilization.cs.dxil.h"
    #include "REBLUR_Perf_Diffuse_TemporalStabilization.cs.dxil.h"
#endif

#ifdef NRD_EMBEDS_SPIRV_SHADERS
    #include "REBLUR_Diffuse_HitDistReconstruction.cs.spirv.h"
    #include "REBLUR_Diffuse_HitDistReconstruction_5x5.cs.spirv.h"
    #include "REBLUR_Diffuse_PrePass.cs.spirv.h"
    #include "REBLUR_Diffuse_TemporalAccumulation.cs.spirv.h"
    #include "REBLUR_Diffuse_HistoryFix.cs.spirv.h"
    #include "REBLUR_Diffuse_Blur.cs.spirv.h"
    #include "REBLUR_Diffuse_TemporalStabilization.cs.spirv.h"
    #include "REBLUR_Diffuse_PostBlur.cs.spirv.h"
    #include "REBLUR_Diffuse_PostBlur_NoTemporalStabilization.cs.spirv.h"
    #include "REBLUR_Diffuse_SplitScreen.cs.spirv.h"

    #include "REBLUR_Perf_Diffuse_HitDistReconstruction.cs.spirv.h"
    #include "REBLUR_Perf_Diffuse_HitDistReconstruction_5x5.cs.spirv.h"
    #include "REBLUR_Perf_Diffuse_PrePass.cs.spirv.h"
    #include "REBLUR_Perf_Diffuse_TemporalAccumulation.cs.spirv.h"
    #include "REBLUR_Perf_Diffuse_HistoryFix.cs.spirv.h"
    #include "REBLUR_Perf_Diffuse_Blur.cs.spirv.h"
    #include "REBLUR_Perf_Diffuse_TemporalStabilization.cs.spirv.h"
    #include "REBLUR_Perf_Diffuse_PostBlur.cs.spirv.h"
    #include "REBLUR_Perf_Diffuse_PostBlur_NoTemporalStabilization.cs.spirv.h"
#endif

#include "Denoisers/Reblur_Diffuse.hpp"

// REBLUR_DIFFUSE_OCCLUSION
#ifdef NRD_EMBEDS_DXBC_SHADERS
    #include "REBLUR_DiffuseOcclusion_HitDistReconstruction.cs.dxbc.h"
    #include "REBLUR_DiffuseOcclusion_HitDistReconstruction_5x5.cs.dxbc.h"
    #include "REBLUR_DiffuseOcclusion_TemporalAccumulation.cs.dxbc.h"
    #include "REBLUR_DiffuseOcclusion_HistoryFix.cs.dxbc.h"
    #include "REBLUR_DiffuseOcclusion_Blur.cs.dxbc.h"
    #include "REBLUR_DiffuseOcclusion_PostBlur_NoTemporalStabilization.cs.dxbc.h"

    #include "REBLUR_Perf_DiffuseOcclusion_HitDistReconstruction.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseOcclusion_HitDistReconstruction_5x5.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseOcclusion_TemporalAccumulation.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseOcclusion_HistoryFix.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseOcclusion_Blur.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseOcclusion_PostBlur_NoTemporalStabilization.cs.dxbc.h"
#endif

#ifdef NRD_EMBEDS_DXIL_SHADERS
    #include "REBLUR_DiffuseOcclusion_HitDistReconstruction.cs.dxil.h"
    #include "REBLUR_DiffuseOcclusion_HitDistReconstruction_5x5.cs.dxil.h"
    #include "REBLUR_DiffuseOcclusion_TemporalAccumulation.cs.dxil.h"
    #include "REBLUR_DiffuseOcclusion_HistoryFix.cs.dxil.h"
    #include "REBLUR_DiffuseOcclusion_Blur.cs.dxil.h"
    #include "REBLUR_DiffuseOcclusion_PostBlur_NoTemporalStabilization.cs.dxil.h"

    #include "REBLUR_Perf_DiffuseOcclusion_HitDistReconstruction.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseOcclusion_HitDistReconstruction_5x5.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseOcclusion_TemporalAccumulation.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseOcclusion_HistoryFix.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseOcclusion_Blur.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseOcclusion_PostBlur_NoTemporalStabilization.cs.dxil.h"
#endif

#ifdef NRD_EMBEDS_SPIRV_SHADERS
    #include "REBLUR_DiffuseOcclusion_HitDistReconstruction.cs.spirv.h"
    #include "REBLUR_DiffuseOcclusion_HitDistReconstruction_5x5.cs.spirv.h"
    #include "REBLUR_DiffuseOcclusion_TemporalAccumulation.cs.spirv.h"
    #include "REBLUR_DiffuseOcclusion_HistoryFix.cs.spirv.h"
    #include "REBLUR_DiffuseOcclusion_Blur.cs.spirv.h"
    #include "REBLUR_DiffuseOcclusion_PostBlur_NoTemporalStabilization.cs.spirv.h"

    #include "REBLUR_Perf_DiffuseOcclusion_HitDistReconstruction.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseOcclusion_HitDistReconstruction_5x5.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseOcclusion_TemporalAccumulation.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseOcclusion_HistoryFix.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseOcclusion_Blur.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseOcclusion_PostBlur_NoTemporalStabilization.cs.spirv.h"
#endif

#include "Denoisers/Reblur_DiffuseOcclusion.hpp"

// REBLUR_DIFFUSE_SH
#ifdef NRD_EMBEDS_DXBC_SHADERS
    #include "REBLUR_DiffuseSh_PrePass.cs.dxbc.h"
    #include "REBLUR_DiffuseSh_TemporalAccumulation.cs.dxbc.h"
    #include "REBLUR_DiffuseSh_HistoryFix.cs.dxbc.h"
    #include "REBLUR_DiffuseSh_Blur.cs.dxbc.h"
    #include "REBLUR_DiffuseSh_PostBlur.cs.dxbc.h"
    #include "REBLUR_DiffuseSh_PostBlur_NoTemporalStabilization.cs.dxbc.h"
    #include "REBLUR_DiffuseSh_TemporalStabilization.cs.dxbc.h"
    #include "REBLUR_DiffuseSh_SplitScreen.cs.dxbc.h"

    #include "REBLUR_Perf_DiffuseSh_PrePass.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSh_TemporalAccumulation.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSh_HistoryFix.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSh_Blur.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSh_PostBlur.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSh_PostBlur_NoTemporalStabilization.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSh_TemporalStabilization.cs.dxbc.h"
#endif

#ifdef NRD_EMBEDS_DXIL_SHADERS
    #include "REBLUR_DiffuseSh_PrePass.cs.dxil.h"
    #include "REBLUR_DiffuseSh_TemporalAccumulation.cs.dxil.h"
    #include "REBLUR_DiffuseSh_HistoryFix.cs.dxil.h"
    #include "REBLUR_DiffuseSh_Blur.cs.dxil.h"
    #include "REBLUR_DiffuseSh_PostBlur.cs.dxil.h"
    #include "REBLUR_DiffuseSh_PostBlur_NoTemporalStabilization.cs.dxil.h"
    #include "REBLUR_DiffuseSh_TemporalStabilization.cs.dxil.h"
    #include "REBLUR_DiffuseSh_SplitScreen.cs.dxil.h"

    #include "REBLUR_Perf_DiffuseSh_PrePass.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSh_TemporalAccumulation.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSh_HistoryFix.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSh_Blur.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSh_PostBlur.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSh_PostBlur_NoTemporalStabilization.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSh_TemporalStabilization.cs.dxil.h"
#endif

#ifdef NRD_EMBEDS_SPIRV_SHADERS
    #include "REBLUR_DiffuseSh_PrePass.cs.spirv.h"
    #include "REBLUR_DiffuseSh_TemporalAccumulation.cs.spirv.h"
    #include "REBLUR_DiffuseSh_HistoryFix.cs.spirv.h"
    #include "REBLUR_DiffuseSh_Blur.cs.spirv.h"
    #include "REBLUR_DiffuseSh_TemporalStabilization.cs.spirv.h"
    #include "REBLUR_DiffuseSh_PostBlur.cs.spirv.h"
    #include "REBLUR_DiffuseSh_PostBlur_NoTemporalStabilization.cs.spirv.h"
    #include "REBLUR_DiffuseSh_SplitScreen.cs.spirv.h"

    #include "REBLUR_Perf_DiffuseSh_PrePass.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSh_TemporalAccumulation.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSh_HistoryFix.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSh_Blur.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSh_TemporalStabilization.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSh_PostBlur.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSh_PostBlur_NoTemporalStabilization.cs.spirv.h"
#endif

#include "Denoisers/Reblur_DiffuseSh.hpp"

// REBLUR_SPECULAR
#ifdef NRD_EMBEDS_DXBC_SHADERS
    #include "REBLUR_Specular_HitDistReconstruction.cs.dxbc.h"
    #include "REBLUR_Specular_HitDistReconstruction_5x5.cs.dxbc.h"
    #include "REBLUR_Specular_PrePass.cs.dxbc.h"
    #include "REBLUR_Specular_TemporalAccumulation.cs.dxbc.h"
    #include "REBLUR_Specular_HistoryFix.cs.dxbc.h"
    #include "REBLUR_Specular_Blur.cs.dxbc.h"
    #include "REBLUR_Specular_PostBlur.cs.dxbc.h"
    #include "REBLUR_Specular_PostBlur_NoTemporalStabilization.cs.dxbc.h"
    #include "REBLUR_Specular_TemporalStabilization.cs.dxbc.h"
    #include "REBLUR_Specular_SplitScreen.cs.dxbc.h"

    #include "REBLUR_Perf_Specular_HitDistReconstruction.cs.dxbc.h"
    #include "REBLUR_Perf_Specular_HitDistReconstruction_5x5.cs.dxbc.h"
    #include "REBLUR_Perf_Specular_PrePass.cs.dxbc.h"
    #include "REBLUR_Perf_Specular_TemporalAccumulation.cs.dxbc.h"
    #include "REBLUR_Perf_Specular_HistoryFix.cs.dxbc.h"
    #include "REBLUR_Perf_Specular_Blur.cs.dxbc.h"
    #include "REBLUR_Perf_Specular_PostBlur.cs.dxbc.h"
    #include "REBLUR_Perf_Specular_PostBlur_NoTemporalStabilization.cs.dxbc.h"
    #include "REBLUR_Perf_Specular_TemporalStabilization.cs.dxbc.h"
#endif

#ifdef NRD_EMBEDS_DXIL_SHADERS
    #include "REBLUR_Specular_HitDistReconstruction.cs.dxil.h"
    #include "REBLUR_Specular_HitDistReconstruction_5x5.cs.dxil.h"
    #include "REBLUR_Specular_PrePass.cs.dxil.h"
    #include "REBLUR_Specular_TemporalAccumulation.cs.dxil.h"
    #include "REBLUR_Specular_HistoryFix.cs.dxil.h"
    #include "REBLUR_Specular_Blur.cs.dxil.h"
    #include "REBLUR_Specular_PostBlur.cs.dxil.h"
    #include "REBLUR_Specular_PostBlur_NoTemporalStabilization.cs.dxil.h"
    #include "REBLUR_Specular_TemporalStabilization.cs.dxil.h"
    #include "REBLUR_Specular_SplitScreen.cs.dxil.h"

    #include "REBLUR_Perf_Specular_HitDistReconstruction.cs.dxil.h"
    #include "REBLUR_Perf_Specular_HitDistReconstruction_5x5.cs.dxil.h"
    #include "REBLUR_Perf_Specular_PrePass.cs.dxil.h"
    #include "REBLUR_Perf_Specular_TemporalAccumulation.cs.dxil.h"
    #include "REBLUR_Perf_Specular_HistoryFix.cs.dxil.h"
    #include "REBLUR_Perf_Specular_Blur.cs.dxil.h"
    #include "REBLUR_Perf_Specular_PostBlur.cs.dxil.h"
    #include "REBLUR_Perf_Specular_PostBlur_NoTemporalStabilization.cs.dxil.h"
    #include "REBLUR_Perf_Specular_TemporalStabilization.cs.dxil.h"

#endif

#ifdef NRD_EMBEDS_SPIRV_SHADERS
    #include "REBLUR_Specular_HitDistReconstruction.cs.spirv.h"
    #include "REBLUR_Specular_HitDistReconstruction_5x5.cs.spirv.h"
    #include "REBLUR_Specular_PrePass.cs.spirv.h"
    #include "REBLUR_Specular_TemporalAccumulation.cs.spirv.h"
    #include "REBLUR_Specular_HistoryFix.cs.spirv.h"
    #include "REBLUR_Specular_Blur.cs.spirv.h"
    #include "REBLUR_Specular_PostBlur.cs.spirv.h"
    #include "REBLUR_Specular_PostBlur_NoTemporalStabilization.cs.spirv.h"
    #include "REBLUR_Specular_TemporalStabilization.cs.spirv.h"
    #include "REBLUR_Specular_SplitScreen.cs.spirv.h"

    #include "REBLUR_Perf_Specular_HitDistReconstruction.cs.spirv.h"
    #include "REBLUR_Perf_Specular_HitDistReconstruction_5x5.cs.spirv.h"
    #include "REBLUR_Perf_Specular_PrePass.cs.spirv.h"
    #include "REBLUR_Perf_Specular_TemporalAccumulation.cs.spirv.h"
    #include "REBLUR_Perf_Specular_HistoryFix.cs.spirv.h"
    #include "REBLUR_Perf_Specular_Blur.cs.spirv.h"
    #include "REBLUR_Perf_Specular_PostBlur.cs.spirv.h"
    #include "REBLUR_Perf_Specular_PostBlur_NoTemporalStabilization.cs.spirv.h"
    #include "REBLUR_Perf_Specular_TemporalStabilization.cs.spirv.h"
#endif

#include "Denoisers/Reblur_Specular.hpp"

// REBLUR_SPECULAR_OCCLUSION
#ifdef NRD_EMBEDS_DXBC_SHADERS
    #include "REBLUR_SpecularOcclusion_HitDistReconstruction.cs.dxbc.h"
    #include "REBLUR_SpecularOcclusion_HitDistReconstruction_5x5.cs.dxbc.h"
    #include "REBLUR_SpecularOcclusion_TemporalAccumulation.cs.dxbc.h"
    #include "REBLUR_SpecularOcclusion_HistoryFix.cs.dxbc.h"
    #include "REBLUR_SpecularOcclusion_Blur.cs.dxbc.h"
    #include "REBLUR_SpecularOcclusion_PostBlur_NoTemporalStabilization.cs.dxbc.h"

    #include "REBLUR_Perf_SpecularOcclusion_HitDistReconstruction.cs.dxbc.h"
    #include "REBLUR_Perf_SpecularOcclusion_HitDistReconstruction_5x5.cs.dxbc.h"
    #include "REBLUR_Perf_SpecularOcclusion_TemporalAccumulation.cs.dxbc.h"
    #include "REBLUR_Perf_SpecularOcclusion_HistoryFix.cs.dxbc.h"
    #include "REBLUR_Perf_SpecularOcclusion_Blur.cs.dxbc.h"
    #include "REBLUR_Perf_SpecularOcclusion_PostBlur_NoTemporalStabilization.cs.dxbc.h"
#endif

#ifdef NRD_EMBEDS_DXIL_SHADERS
    #include "REBLUR_SpecularOcclusion_HitDistReconstruction.cs.dxil.h"
    #include "REBLUR_SpecularOcclusion_HitDistReconstruction_5x5.cs.dxil.h"
    #include "REBLUR_SpecularOcclusion_TemporalAccumulation.cs.dxil.h"
    #include "REBLUR_SpecularOcclusion_HistoryFix.cs.dxil.h"
    #include "REBLUR_SpecularOcclusion_Blur.cs.dxil.h"
    #include "REBLUR_SpecularOcclusion_PostBlur_NoTemporalStabilization.cs.dxil.h"

    #include "REBLUR_Perf_SpecularOcclusion_HitDistReconstruction.cs.dxil.h"
    #include "REBLUR_Perf_SpecularOcclusion_HitDistReconstruction_5x5.cs.dxil.h"
    #include "REBLUR_Perf_SpecularOcclusion_TemporalAccumulation.cs.dxil.h"
    #include "REBLUR_Perf_SpecularOcclusion_HistoryFix.cs.dxil.h"
    #include "REBLUR_Perf_SpecularOcclusion_Blur.cs.dxil.h"
    #include "REBLUR_Perf_SpecularOcclusion_PostBlur_NoTemporalStabilization.cs.dxil.h"
#endif

#ifdef NRD_EMBEDS_SPIRV_SHADERS
    #include "REBLUR_SpecularOcclusion_HitDistReconstruction.cs.spirv.h"
    #include "REBLUR_SpecularOcclusion_HitDistReconstruction_5x5.cs.spirv.h"
    #include "REBLUR_SpecularOcclusion_TemporalAccumulation.cs.spirv.h"
    #include "REBLUR_SpecularOcclusion_HistoryFix.cs.spirv.h"
    #include "REBLUR_SpecularOcclusion_Blur.cs.spirv.h"
    #include "REBLUR_SpecularOcclusion_PostBlur_NoTemporalStabilization.cs.spirv.h"

    #include "REBLUR_Perf_SpecularOcclusion_HitDistReconstruction.cs.spirv.h"
    #include "REBLUR_Perf_SpecularOcclusion_HitDistReconstruction_5x5.cs.spirv.h"
    #include "REBLUR_Perf_SpecularOcclusion_TemporalAccumulation.cs.spirv.h"
    #include "REBLUR_Perf_SpecularOcclusion_HistoryFix.cs.spirv.h"
    #include "REBLUR_Perf_SpecularOcclusion_Blur.cs.spirv.h"
    #include "REBLUR_Perf_SpecularOcclusion_PostBlur_NoTemporalStabilization.cs.spirv.h"
#endif

#include "Denoisers/Reblur_SpecularOcclusion.hpp"

// REBLUR_SPECULAR_SH
#ifdef NRD_EMBEDS_DXBC_SHADERS
    #include "REBLUR_SpecularSh_PrePass.cs.dxbc.h"
    #include "REBLUR_SpecularSh_TemporalAccumulation.cs.dxbc.h"
    #include "REBLUR_SpecularSh_HistoryFix.cs.dxbc.h"
    #include "REBLUR_SpecularSh_Blur.cs.dxbc.h"
    #include "REBLUR_SpecularSh_PostBlur.cs.dxbc.h"
    #include "REBLUR_SpecularSh_PostBlur_NoTemporalStabilization.cs.dxbc.h"
    #include "REBLUR_SpecularSh_TemporalStabilization.cs.dxbc.h"
    #include "REBLUR_SpecularSh_SplitScreen.cs.dxbc.h"

    #include "REBLUR_Perf_SpecularSh_PrePass.cs.dxbc.h"
    #include "REBLUR_Perf_SpecularSh_TemporalAccumulation.cs.dxbc.h"
    #include "REBLUR_Perf_SpecularSh_HistoryFix.cs.dxbc.h"
    #include "REBLUR_Perf_SpecularSh_Blur.cs.dxbc.h"
    #include "REBLUR_Perf_SpecularSh_PostBlur.cs.dxbc.h"
    #include "REBLUR_Perf_SpecularSh_PostBlur_NoTemporalStabilization.cs.dxbc.h"
    #include "REBLUR_Perf_SpecularSh_TemporalStabilization.cs.dxbc.h"
#endif

#ifdef NRD_EMBEDS_DXIL_SHADERS
    #include "REBLUR_SpecularSh_PrePass.cs.dxil.h"
    #include "REBLUR_SpecularSh_TemporalAccumulation.cs.dxil.h"
    #include "REBLUR_SpecularSh_HistoryFix.cs.dxil.h"
    #include "REBLUR_SpecularSh_Blur.cs.dxil.h"
    #include "REBLUR_SpecularSh_PostBlur.cs.dxil.h"
    #include "REBLUR_SpecularSh_PostBlur_NoTemporalStabilization.cs.dxil.h"
    #include "REBLUR_SpecularSh_TemporalStabilization.cs.dxil.h"
    #include "REBLUR_SpecularSh_SplitScreen.cs.dxil.h"

    #include "REBLUR_Perf_SpecularSh_PrePass.cs.dxil.h"
    #include "REBLUR_Perf_SpecularSh_TemporalAccumulation.cs.dxil.h"
    #include "REBLUR_Perf_SpecularSh_HistoryFix.cs.dxil.h"
    #include "REBLUR_Perf_SpecularSh_Blur.cs.dxil.h"
    #include "REBLUR_Perf_SpecularSh_PostBlur.cs.dxil.h"
    #include "REBLUR_Perf_SpecularSh_PostBlur_NoTemporalStabilization.cs.dxil.h"
    #include "REBLUR_Perf_SpecularSh_TemporalStabilization.cs.dxil.h"
#endif

#ifdef NRD_EMBEDS_SPIRV_SHADERS
    #include "REBLUR_SpecularSh_PrePass.cs.spirv.h"
    #include "REBLUR_SpecularSh_TemporalAccumulation.cs.spirv.h"
    #include "REBLUR_SpecularSh_HistoryFix.cs.spirv.h"
    #include "REBLUR_SpecularSh_Blur.cs.spirv.h"
    #include "REBLUR_SpecularSh_PostBlur.cs.spirv.h"
    #include "REBLUR_SpecularSh_PostBlur_NoTemporalStabilization.cs.spirv.h"
    #include "REBLUR_SpecularSh_TemporalStabilization.cs.spirv.h"
    #include "REBLUR_SpecularSh_SplitScreen.cs.spirv.h"

    #include "REBLUR_Perf_SpecularSh_PrePass.cs.spirv.h"
    #include "REBLUR_Perf_SpecularSh_TemporalAccumulation.cs.spirv.h"
    #include "REBLUR_Perf_SpecularSh_HistoryFix.cs.spirv.h"
    #include "REBLUR_Perf_SpecularSh_Blur.cs.spirv.h"
    #include "REBLUR_Perf_SpecularSh_PostBlur.cs.spirv.h"
    #include "REBLUR_Perf_SpecularSh_PostBlur_NoTemporalStabilization.cs.spirv.h"
    #include "REBLUR_Perf_SpecularSh_TemporalStabilization.cs.spirv.h"
#endif

#include "Denoisers/Reblur_SpecularSh.hpp"

// REBLUR_DIFFUSE_SPECULAR
#ifdef NRD_EMBEDS_DXBC_SHADERS
    #include "REBLUR_DiffuseSpecular_HitDistReconstruction.cs.dxbc.h"
    #include "REBLUR_DiffuseSpecular_HitDistReconstruction_5x5.cs.dxbc.h"
    #include "REBLUR_DiffuseSpecular_PrePass.cs.dxbc.h"
    #include "REBLUR_DiffuseSpecular_TemporalAccumulation.cs.dxbc.h"
    #include "REBLUR_DiffuseSpecular_HistoryFix.cs.dxbc.h"
    #include "REBLUR_DiffuseSpecular_Blur.cs.dxbc.h"
    #include "REBLUR_DiffuseSpecular_TemporalStabilization.cs.dxbc.h"
    #include "REBLUR_DiffuseSpecular_PostBlur.cs.dxbc.h"
    #include "REBLUR_DiffuseSpecular_PostBlur_NoTemporalStabilization.cs.dxbc.h"
    #include "REBLUR_DiffuseSpecular_SplitScreen.cs.dxbc.h"

    #include "REBLUR_Perf_DiffuseSpecular_HitDistReconstruction.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSpecular_HitDistReconstruction_5x5.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSpecular_PrePass.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSpecular_TemporalAccumulation.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSpecular_HistoryFix.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSpecular_Blur.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSpecular_TemporalStabilization.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSpecular_PostBlur.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSpecular_PostBlur_NoTemporalStabilization.cs.dxbc.h"
#endif

#ifdef NRD_EMBEDS_DXIL_SHADERS
    #include "REBLUR_DiffuseSpecular_HitDistReconstruction.cs.dxil.h"
    #include "REBLUR_DiffuseSpecular_HitDistReconstruction_5x5.cs.dxil.h"
    #include "REBLUR_DiffuseSpecular_PrePass.cs.dxil.h"
    #include "REBLUR_DiffuseSpecular_TemporalAccumulation.cs.dxil.h"
    #include "REBLUR_DiffuseSpecular_HistoryFix.cs.dxil.h"
    #include "REBLUR_DiffuseSpecular_Blur.cs.dxil.h"
    #include "REBLUR_DiffuseSpecular_TemporalStabilization.cs.dxil.h"
    #include "REBLUR_DiffuseSpecular_PostBlur.cs.dxil.h"
    #include "REBLUR_DiffuseSpecular_PostBlur_NoTemporalStabilization.cs.dxil.h"
    #include "REBLUR_DiffuseSpecular_SplitScreen.cs.dxil.h"

    #include "REBLUR_Perf_DiffuseSpecular_HitDistReconstruction.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSpecular_HitDistReconstruction_5x5.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSpecular_PrePass.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSpecular_TemporalAccumulation.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSpecular_HistoryFix.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSpecular_Blur.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSpecular_TemporalStabilization.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSpecular_PostBlur.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSpecular_PostBlur_NoTemporalStabilization.cs.dxil.h"
#endif

#ifdef NRD_EMBEDS_SPIRV_SHADERS
    #include "REBLUR_DiffuseSpecular_HitDistReconstruction.cs.spirv.h"
    #include "REBLUR_DiffuseSpecular_HitDistReconstruction_5x5.cs.spirv.h"
    #include "REBLUR_DiffuseSpecular_PrePass.cs.spirv.h"
    #include "REBLUR_DiffuseSpecular_TemporalAccumulation.cs.spirv.h"
    #include "REBLUR_DiffuseSpecular_HistoryFix.cs.spirv.h"
    #include "REBLUR_DiffuseSpecular_Blur.cs.spirv.h"
    #include "REBLUR_DiffuseSpecular_TemporalStabilization.cs.spirv.h"
    #include "REBLUR_DiffuseSpecular_PostBlur.cs.spirv.h"
    #include "REBLUR_DiffuseSpecular_PostBlur_NoTemporalStabilization.cs.spirv.h"
    #include "REBLUR_DiffuseSpecular_SplitScreen.cs.spirv.h"

    #include "REBLUR_Perf_DiffuseSpecular_HitDistReconstruction.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSpecular_HitDistReconstruction_5x5.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSpecular_PrePass.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSpecular_TemporalAccumulation.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSpecular_HistoryFix.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSpecular_Blur.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSpecular_TemporalStabilization.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSpecular_PostBlur.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSpecular_PostBlur_NoTemporalStabilization.cs.spirv.h"
#endif

#include "Denoisers/Reblur_DiffuseSpecular.hpp"

// REBLUR_DIFFUSE_SPECULAR_OCCLUSION
#ifdef NRD_EMBEDS_DXBC_SHADERS
    #include "REBLUR_DiffuseSpecularOcclusion_HitDistReconstruction.cs.dxbc.h"
    #include "REBLUR_DiffuseSpecularOcclusion_HitDistReconstruction_5x5.cs.dxbc.h"
    #include "REBLUR_DiffuseSpecularOcclusion_TemporalAccumulation.cs.dxbc.h"
    #include "REBLUR_DiffuseSpecularOcclusion_HistoryFix.cs.dxbc.h"
    #include "REBLUR_DiffuseSpecularOcclusion_Blur.cs.dxbc.h"
    #include "REBLUR_DiffuseSpecularOcclusion_PostBlur_NoTemporalStabilization.cs.dxbc.h"

    #include "REBLUR_Perf_DiffuseSpecularOcclusion_HitDistReconstruction.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSpecularOcclusion_HitDistReconstruction_5x5.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSpecularOcclusion_TemporalAccumulation.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSpecularOcclusion_HistoryFix.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSpecularOcclusion_Blur.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSpecularOcclusion_PostBlur_NoTemporalStabilization.cs.dxbc.h"
#endif

#ifdef NRD_EMBEDS_DXIL_SHADERS
    #include "REBLUR_DiffuseSpecularOcclusion_HitDistReconstruction.cs.dxil.h"
    #include "REBLUR_DiffuseSpecularOcclusion_HitDistReconstruction_5x5.cs.dxil.h"
    #include "REBLUR_DiffuseSpecularOcclusion_TemporalAccumulation.cs.dxil.h"
    #include "REBLUR_DiffuseSpecularOcclusion_HistoryFix.cs.dxil.h"
    #include "REBLUR_DiffuseSpecularOcclusion_Blur.cs.dxil.h"
    #include "REBLUR_DiffuseSpecularOcclusion_PostBlur_NoTemporalStabilization.cs.dxil.h"

    #include "REBLUR_Perf_DiffuseSpecularOcclusion_HitDistReconstruction.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSpecularOcclusion_HitDistReconstruction_5x5.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSpecularOcclusion_TemporalAccumulation.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSpecularOcclusion_HistoryFix.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSpecularOcclusion_Blur.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSpecularOcclusion_PostBlur_NoTemporalStabilization.cs.dxil.h"
#endif

#ifdef NRD_EMBEDS_SPIRV_SHADERS
    #include "REBLUR_DiffuseSpecularOcclusion_HitDistReconstruction.cs.spirv.h"
    #include "REBLUR_DiffuseSpecularOcclusion_HitDistReconstruction_5x5.cs.spirv.h"
    #include "REBLUR_DiffuseSpecularOcclusion_TemporalAccumulation.cs.spirv.h"
    #include "REBLUR_DiffuseSpecularOcclusion_HistoryFix.cs.spirv.h"
    #include "REBLUR_DiffuseSpecularOcclusion_Blur.cs.spirv.h"
    #include "REBLUR_DiffuseSpecularOcclusion_PostBlur_NoTemporalStabilization.cs.spirv.h"

    #include "REBLUR_Perf_DiffuseSpecularOcclusion_HitDistReconstruction.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSpecularOcclusion_HitDistReconstruction_5x5.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSpecularOcclusion_TemporalAccumulation.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSpecularOcclusion_HistoryFix.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSpecularOcclusion_Blur.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSpecularOcclusion_PostBlur_NoTemporalStabilization.cs.spirv.h"
#endif

#include "Denoisers/Reblur_DiffuseSpecularOcclusion.hpp"

// REBLUR_DIFFUSE_SPECULAR_SH
#ifdef NRD_EMBEDS_DXBC_SHADERS
    #include "REBLUR_DiffuseSpecularSh_PrePass.cs.dxbc.h"
    #include "REBLUR_DiffuseSpecularSh_TemporalAccumulation.cs.dxbc.h"
    #include "REBLUR_DiffuseSpecularSh_HistoryFix.cs.dxbc.h"
    #include "REBLUR_DiffuseSpecularSh_Blur.cs.dxbc.h"
    #include "REBLUR_DiffuseSpecularSh_TemporalStabilization.cs.dxbc.h"
    #include "REBLUR_DiffuseSpecularSh_PostBlur.cs.dxbc.h"
    #include "REBLUR_DiffuseSpecularSh_PostBlur_NoTemporalStabilization.cs.dxbc.h"
    #include "REBLUR_DiffuseSpecularSh_SplitScreen.cs.dxbc.h"

    #include "REBLUR_Perf_DiffuseSpecularSh_PrePass.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSpecularSh_TemporalAccumulation.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSpecularSh_HistoryFix.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSpecularSh_Blur.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSpecularSh_TemporalStabilization.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSpecularSh_PostBlur.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseSpecularSh_PostBlur_NoTemporalStabilization.cs.dxbc.h"
#endif

#ifdef NRD_EMBEDS_DXIL_SHADERS
    #include "REBLUR_DiffuseSpecularSh_PrePass.cs.dxil.h"
    #include "REBLUR_DiffuseSpecularSh_TemporalAccumulation.cs.dxil.h"
    #include "REBLUR_DiffuseSpecularSh_HistoryFix.cs.dxil.h"
    #include "REBLUR_DiffuseSpecularSh_Blur.cs.dxil.h"
    #include "REBLUR_DiffuseSpecularSh_TemporalStabilization.cs.dxil.h"
    #include "REBLUR_DiffuseSpecularSh_PostBlur.cs.dxil.h"
    #include "REBLUR_DiffuseSpecularSh_PostBlur_NoTemporalStabilization.cs.dxil.h"
    #include "REBLUR_DiffuseSpecularSh_SplitScreen.cs.dxil.h"

    #include "REBLUR_Perf_DiffuseSpecularSh_PrePass.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSpecularSh_TemporalAccumulation.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSpecularSh_HistoryFix.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSpecularSh_Blur.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSpecularSh_TemporalStabilization.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSpecularSh_PostBlur.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseSpecularSh_PostBlur_NoTemporalStabilization.cs.dxil.h"
#endif

#ifdef NRD_EMBEDS_SPIRV_SHADERS
    #include "REBLUR_DiffuseSpecularSh_PrePass.cs.spirv.h"
    #include "REBLUR_DiffuseSpecularSh_TemporalAccumulation.cs.spirv.h"
    #include "REBLUR_DiffuseSpecularSh_HistoryFix.cs.spirv.h"
    #include "REBLUR_DiffuseSpecularSh_Blur.cs.spirv.h"
    #include "REBLUR_DiffuseSpecularSh_TemporalStabilization.cs.spirv.h"
    #include "REBLUR_DiffuseSpecularSh_PostBlur.cs.spirv.h"
    #include "REBLUR_DiffuseSpecularSh_PostBlur_NoTemporalStabilization.cs.spirv.h"
    #include "REBLUR_DiffuseSpecularSh_SplitScreen.cs.spirv.h"

    #include "REBLUR_Perf_DiffuseSpecularSh_PrePass.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSpecularSh_TemporalAccumulation.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSpecularSh_HistoryFix.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSpecularSh_Blur.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSpecularSh_TemporalStabilization.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSpecularSh_PostBlur.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseSpecularSh_PostBlur_NoTemporalStabilization.cs.spirv.h"
#endif

#include "Denoisers/Reblur_DiffuseSpecularSh.hpp"

// REBLUR_DIFFUSE_DIRECTIONAL_OCCLUSION
#ifdef NRD_EMBEDS_DXBC_SHADERS
    #include "REBLUR_DiffuseDirectionalOcclusion_PrePass.cs.dxbc.h"
    #include "REBLUR_DiffuseDirectionalOcclusion_TemporalAccumulation.cs.dxbc.h"
    #include "REBLUR_DiffuseDirectionalOcclusion_HistoryFix.cs.dxbc.h"
    #include "REBLUR_DiffuseDirectionalOcclusion_Blur.cs.dxbc.h"
    #include "REBLUR_DiffuseDirectionalOcclusion_PostBlur.cs.dxbc.h"
    #include "REBLUR_DiffuseDirectionalOcclusion_PostBlur_NoTemporalStabilization.cs.dxbc.h"
    #include "REBLUR_DiffuseDirectionalOcclusion_TemporalStabilization.cs.dxbc.h"

    #include "REBLUR_Perf_DiffuseDirectionalOcclusion_PrePass.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseDirectionalOcclusion_TemporalAccumulation.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseDirectionalOcclusion_HistoryFix.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseDirectionalOcclusion_Blur.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseDirectionalOcclusion_PostBlur.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseDirectionalOcclusion_PostBlur_NoTemporalStabilization.cs.dxbc.h"
    #include "REBLUR_Perf_DiffuseDirectionalOcclusion_TemporalStabilization.cs.dxbc.h"
#endif

#ifdef NRD_EMBEDS_DXIL_SHADERS
    #include "REBLUR_DiffuseDirectionalOcclusion_PrePass.cs.dxil.h"
    #include "REBLUR_DiffuseDirectionalOcclusion_TemporalAccumulation.cs.dxil.h"
    #include "REBLUR_DiffuseDirectionalOcclusion_HistoryFix.cs.dxil.h"
    #include "REBLUR_DiffuseDirectionalOcclusion_Blur.cs.dxil.h"
    #include "REBLUR_DiffuseDirectionalOcclusion_PostBlur.cs.dxil.h"
    #include "REBLUR_DiffuseDirectionalOcclusion_PostBlur_NoTemporalStabilization.cs.dxil.h"
    #include "REBLUR_DiffuseDirectionalOcclusion_TemporalStabilization.cs.dxil.h"

    #include "REBLUR_Perf_DiffuseDirectionalOcclusion_PrePass.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseDirectionalOcclusion_TemporalAccumulation.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseDirectionalOcclusion_HistoryFix.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseDirectionalOcclusion_Blur.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseDirectionalOcclusion_PostBlur.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseDirectionalOcclusion_PostBlur_NoTemporalStabilization.cs.dxil.h"
    #include "REBLUR_Perf_DiffuseDirectionalOcclusion_TemporalStabilization.cs.dxil.h"
#endif

#ifdef NRD_EMBEDS_SPIRV_SHADERS
    #include "REBLUR_DiffuseDirectionalOcclusion_PrePass.cs.spirv.h"
    #include "REBLUR_DiffuseDirectionalOcclusion_TemporalAccumulation.cs.spirv.h"
    #include "REBLUR_DiffuseDirectionalOcclusion_HistoryFix.cs.spirv.h"
    #include "REBLUR_DiffuseDirectionalOcclusion_Blur.cs.spirv.h"
    #include "REBLUR_DiffuseDirectionalOcclusion_TemporalStabilization.cs.spirv.h"
    #include "REBLUR_DiffuseDirectionalOcclusion_PostBlur.cs.spirv.h"
    #include "REBLUR_DiffuseDirectionalOcclusion_PostBlur_NoTemporalStabilization.cs.spirv.h"

    #include "REBLUR_Perf_DiffuseDirectionalOcclusion_PrePass.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseDirectionalOcclusion_TemporalAccumulation.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseDirectionalOcclusion_HistoryFix.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseDirectionalOcclusion_Blur.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseDirectionalOcclusion_TemporalStabilization.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseDirectionalOcclusion_PostBlur.cs.spirv.h"
    #include "REBLUR_Perf_DiffuseDirectionalOcclusion_PostBlur_NoTemporalStabilization.cs.spirv.h"
#endif

#include "Denoisers/Reblur_DiffuseDirectionalOcclusion.hpp"
