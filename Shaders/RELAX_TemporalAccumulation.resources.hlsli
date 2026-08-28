/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

NRD_CONSTANTS_START( RELAX_TemporalAccumulationConstants )
    RELAX_SHARED_CONSTANTS
NRD_CONSTANTS_END

NRD_SAMPLERS_START
    NRD_SAMPLER( SamplerState, gNearestClamp, s, 0 )
    NRD_SAMPLER( SamplerState, gLinearClamp, s, 1 )
NRD_SAMPLERS_END

NRD_INPUTS_START
    NRD_INPUT( Texture2D, float, gIn_Tiles, t, 0, NRD_RESOURCE_TRANSIENT )
    NRD_INPUT( Texture2D, float3, gIn_Mv, t, 1, NRD_RESOURCE_IN )
    NRD_INPUT( Texture2D, float4, gIn_Normal_Roughness, t, 2, NRD_RESOURCE_IN )
    NRD_INPUT( Texture2D, float, gIn_ViewZ, t, 3, NRD_RESOURCE_IN )
    NRD_INPUT( Texture2D, float, gIn_DisocclusionThresholdMix, t, 4, NRD_RESOURCE_IN_ZERO_OFFSET )
    NRD_INPUT( Texture2D, float4, gPrev_Normal_Roughness, t, 5, NRD_RESOURCE_PERMANENT )
    NRD_INPUT( Texture2D, float, gPrev_ViewZ, t, 6, NRD_RESOURCE_PERMANENT )
    NRD_INPUT( Texture2D, float, gPrev_HistoryLength, t, 7, NRD_RESOURCE_PERMANENT )
    NRD_INPUT( Texture2D, float, gPrev_MateriallID, t, 8, NRD_RESOURCE_PERMANENT )
    #if( NRD_HAS_DIFF && NRD_HAS_SPEC )
        NRD_INPUT( Texture2D, float4, gIn_Spec, t, 9, NRD_RESOURCE_OUT )
        NRD_INPUT( Texture2D, float4, gIn_Diff, t, 10, NRD_RESOURCE_OUT )
        NRD_INPUT( Texture2D, float4, gHistory_SpecFast, t, 11, NRD_RESOURCE_PERMANENT )
        NRD_INPUT( Texture2D, float4, gHistory_DiffFast, t, 12, NRD_RESOURCE_PERMANENT )
        NRD_INPUT( Texture2D, float4, gHistory_Spec, t, 13, NRD_RESOURCE_PERMANENT )
        NRD_INPUT( Texture2D, float4, gHistory_Diff, t, 14, NRD_RESOURCE_PERMANENT )
        NRD_INPUT( Texture2D, float, gPrev_SpecHitDist, t, 15, NRD_RESOURCE_PERMANENT )
        NRD_INPUT( Texture2D, float, gIn_SpecConfidence, t, 16, NRD_RESOURCE_IN_ZERO_OFFSET )
        NRD_INPUT( Texture2D, float, gIn_DiffConfidence, t, 17, NRD_RESOURCE_IN_ZERO_OFFSET )
        #if( NRD_MODE == NRD_MODE_SH )
            NRD_INPUT( Texture2D, RELAX_SH_TYPE, gIn_SpecSh, t, 18, NRD_RESOURCE_OUT )
            NRD_INPUT( Texture2D, RELAX_SH_TYPE, gIn_DiffSh, t, 19, NRD_RESOURCE_OUT )
            NRD_INPUT( Texture2D, RELAX_SH_TYPE, gHistory_SpecShFast, t, 20, NRD_RESOURCE_PERMANENT )
            NRD_INPUT( Texture2D, RELAX_SH_TYPE, gHistory_DiffShFast, t, 21, NRD_RESOURCE_PERMANENT )
            NRD_INPUT( Texture2D, RELAX_SH_TYPE, gHistory_SpecSh, t, 22, NRD_RESOURCE_PERMANENT )
            NRD_INPUT( Texture2D, RELAX_SH_TYPE, gHistory_DiffSh, t, 23, NRD_RESOURCE_PERMANENT )
        #endif
    #elif( NRD_HAS_DIFF )
        NRD_INPUT( Texture2D, float4, gIn_Diff, t, 9, NRD_RESOURCE_OUT )
        NRD_INPUT( Texture2D, float4, gHistory_DiffFast, t, 10, NRD_RESOURCE_PERMANENT )
        NRD_INPUT( Texture2D, float4, gHistory_Diff, t, 11, NRD_RESOURCE_PERMANENT )
        NRD_INPUT( Texture2D, float, gIn_DiffConfidence, t, 12, NRD_RESOURCE_IN_ZERO_OFFSET )
        #if( NRD_MODE == NRD_MODE_SH )
            NRD_INPUT( Texture2D, RELAX_SH_TYPE, gIn_DiffSh, t, 13, NRD_RESOURCE_OUT )
            NRD_INPUT( Texture2D, RELAX_SH_TYPE, gHistory_DiffShFast, t, 14, NRD_RESOURCE_PERMANENT )
            NRD_INPUT( Texture2D, RELAX_SH_TYPE, gHistory_DiffSh, t, 15, NRD_RESOURCE_PERMANENT )
        #endif
    #else
        NRD_INPUT( Texture2D, float4, gIn_Spec, t, 9, NRD_RESOURCE_OUT )
        NRD_INPUT( Texture2D, float4, gHistory_SpecFast, t, 10, NRD_RESOURCE_PERMANENT )
        NRD_INPUT( Texture2D, float4, gHistory_Spec, t, 11, NRD_RESOURCE_PERMANENT )
        NRD_INPUT( Texture2D, float, gPrev_SpecHitDist, t, 12, NRD_RESOURCE_PERMANENT )
        NRD_INPUT( Texture2D, float, gIn_SpecConfidence, t, 13, NRD_RESOURCE_IN_ZERO_OFFSET )
        #if( NRD_MODE == NRD_MODE_SH )
            NRD_INPUT( Texture2D, RELAX_SH_TYPE, gIn_SpecSh, t, 14, NRD_RESOURCE_OUT )
            NRD_INPUT( Texture2D, RELAX_SH_TYPE, gHistory_SpecShFast, t, 15, NRD_RESOURCE_PERMANENT )
            NRD_INPUT( Texture2D, RELAX_SH_TYPE, gHistory_SpecSh, t, 16, NRD_RESOURCE_PERMANENT )
        #endif
    #endif
NRD_INPUTS_END

NRD_OUTPUTS_START
    NRD_OUTPUT( RWTexture2D, float, gOut_HistoryLength, u, 0, NRD_RESOURCE_TRANSIENT )
    #if( NRD_HAS_DIFF && NRD_HAS_SPEC )
        NRD_OUTPUT( RWTexture2D, float4, gOut_Spec, u, 1, NRD_RESOURCE_TRANSIENT )
        NRD_OUTPUT( RWTexture2D, float4, gOut_Diff, u, 2, NRD_RESOURCE_TRANSIENT )
        NRD_OUTPUT( RWTexture2D, float4, gOut_SpecFast, u, 3, NRD_RESOURCE_TRANSIENT )
        NRD_OUTPUT( RWTexture2D, float4, gOut_DiffFast, u, 4, NRD_RESOURCE_TRANSIENT )
        NRD_OUTPUT( RWTexture2D, float, gOut_SpecHitDist, u, 5, NRD_RESOURCE_TRANSIENT )
        NRD_OUTPUT( RWTexture2D, float, gOut_SpecReprojectionConfidence, u, 6, NRD_RESOURCE_TRANSIENT )
        #if( NRD_MODE == NRD_MODE_SH )
            NRD_OUTPUT( RWTexture2D, RELAX_SH_TYPE, gOut_SpecSh, u, 7, NRD_RESOURCE_TRANSIENT )
            NRD_OUTPUT( RWTexture2D, RELAX_SH_TYPE, gOut_DiffSh, u, 8, NRD_RESOURCE_TRANSIENT )
            NRD_OUTPUT( RWTexture2D, RELAX_SH_TYPE, gOut_SpecShFast, u, 9, NRD_RESOURCE_TRANSIENT )
            NRD_OUTPUT( RWTexture2D, RELAX_SH_TYPE, gOut_DiffShFast, u, 10, NRD_RESOURCE_TRANSIENT )
        #endif
    #elif( NRD_HAS_DIFF )
        NRD_OUTPUT( RWTexture2D, float4, gOut_Diff, u, 1, NRD_RESOURCE_TRANSIENT )
        NRD_OUTPUT( RWTexture2D, float4, gOut_DiffFast, u, 2, NRD_RESOURCE_TRANSIENT )
        #if( NRD_MODE == NRD_MODE_SH )
            NRD_OUTPUT( RWTexture2D, RELAX_SH_TYPE, gOut_DiffSh, u, 3, NRD_RESOURCE_TRANSIENT )
            NRD_OUTPUT( RWTexture2D, RELAX_SH_TYPE, gOut_DiffShFast, u, 4, NRD_RESOURCE_TRANSIENT )
        #endif
    #else
        NRD_OUTPUT( RWTexture2D, float4, gOut_Spec, u, 1, NRD_RESOURCE_TRANSIENT )
        NRD_OUTPUT( RWTexture2D, float4, gOut_SpecFast, u, 2, NRD_RESOURCE_TRANSIENT )
        NRD_OUTPUT( RWTexture2D, float, gOut_SpecHitDist, u, 3, NRD_RESOURCE_TRANSIENT )
        NRD_OUTPUT( RWTexture2D, float, gOut_SpecReprojectionConfidence, u, 4, NRD_RESOURCE_TRANSIENT )
        #if( NRD_MODE == NRD_MODE_SH )
            NRD_OUTPUT( RWTexture2D, RELAX_SH_TYPE, gOut_SpecSh, u, 5, NRD_RESOURCE_TRANSIENT )
            NRD_OUTPUT( RWTexture2D, RELAX_SH_TYPE, gOut_SpecShFast, u, 6, NRD_RESOURCE_TRANSIENT )
        #endif
    #endif
NRD_OUTPUTS_END

// Macro magic
#define RELAX_TemporalAccumulationGroupX 8
#define RELAX_TemporalAccumulationGroupY 16

// Shader only
#ifndef __cplusplus

#define NRD_BORDER 1

#define GROUP_X RELAX_TemporalAccumulationGroupX
#define GROUP_Y RELAX_TemporalAccumulationGroupY

#endif
