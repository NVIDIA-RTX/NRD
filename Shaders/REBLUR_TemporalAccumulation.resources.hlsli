/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

NRD_CONSTANTS_START( REBLUR_TemporalAccumulationConstants )
    REBLUR_SHARED_CONSTANTS
NRD_CONSTANTS_END

NRD_SAMPLERS_START
    NRD_SAMPLER( SamplerState, gNearestClamp, s, 0 )
    NRD_SAMPLER( SamplerState, gLinearClamp, s, 1 )
NRD_SAMPLERS_END

NRD_INPUTS_START
    NRD_INPUT( Texture2D, REBLUR_TILE_TYPE, gIn_Tiles, t, 0, NRD_RESOURCE_TRANSIENT )
    NRD_INPUT( Texture2D, float4, gIn_Normal_Roughness, t, 1, NRD_RESOURCE_IN )
    NRD_INPUT( Texture2D, float, gIn_ViewZ, t, 2, NRD_RESOURCE_IN )
    NRD_INPUT( Texture2D, float3, gIn_Mv, t, 3, NRD_RESOURCE_IN )
    NRD_INPUT( Texture2D, float, gPrev_ViewZ, t, 4, NRD_RESOURCE_PERMANENT )
    NRD_INPUT( Texture2D, float4, gPrev_Normal_Roughness, t, 5, NRD_RESOURCE_PERMANENT )
    NRD_INPUT( Texture2D, uint, gPrev_InternalData, t, 6, NRD_RESOURCE_PERMANENT )
    NRD_INPUT( Texture2D, float, gIn_DisocclusionThresholdMix, t, 7, NRD_RESOURCE_IN_ZERO_OFFSET )
    #if( NRD_HAS_DIFF && NRD_HAS_SPEC )
        NRD_INPUT( Texture2D, float, gIn_DiffConfidence, t, 8, NRD_RESOURCE_IN_ZERO_OFFSET )
        NRD_INPUT( Texture2D, float, gIn_SpecConfidence, t, 9, NRD_RESOURCE_IN_ZERO_OFFSET )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gIn_Diff, t, 10, NRD_RESOURCE_IN_DISPATCH )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gIn_Spec, t, 11, NRD_RESOURCE_IN_DISPATCH )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gHistory_Diff, t, 12, NRD_RESOURCE_PERMANENT )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gHistory_Spec, t, 13, NRD_RESOURCE_PERMANENT )
        NRD_INPUT( Texture2D, REBLUR_FAST_TYPE, gHistory_DiffFast, t, 14, NRD_RESOURCE_PERMANENT )
        NRD_INPUT( Texture2D, REBLUR_FAST_TYPE, gHistory_SpecFast, t, 15, NRD_RESOURCE_PERMANENT )
        NRD_INPUT( Texture2D, float, gPrev_SpecHitDistForTracking, t, 16, NRD_RESOURCE_PERMANENT )
        #if( NRD_MODE != NRD_MODE_OCCLUSION )
            NRD_INPUT( Texture2D, float, gIn_SpecHitDistForTracking, t, 17, NRD_RESOURCE_TRANSIENT )
        #endif
        #if( NRD_MODE == NRD_MODE_SH )
            NRD_INPUT( Texture2D, REBLUR_SH_TYPE, gIn_DiffSh, t, 18, NRD_RESOURCE_IN_DISPATCH )
            NRD_INPUT( Texture2D, REBLUR_SH_TYPE, gIn_SpecSh, t, 19, NRD_RESOURCE_IN_DISPATCH )
            NRD_INPUT( Texture2D, REBLUR_SH_TYPE, gHistory_DiffSh, t, 20, NRD_RESOURCE_PERMANENT )
            NRD_INPUT( Texture2D, REBLUR_SH_TYPE, gHistory_SpecSh, t, 21, NRD_RESOURCE_PERMANENT )
        #endif
    #elif( NRD_HAS_DIFF )
        NRD_INPUT( Texture2D, float, gIn_DiffConfidence, t, 8, NRD_RESOURCE_IN_ZERO_OFFSET )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gIn_Diff, t, 9, NRD_RESOURCE_IN_DISPATCH )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gHistory_Diff, t, 10, NRD_RESOURCE_PERMANENT )
        NRD_INPUT( Texture2D, REBLUR_FAST_TYPE, gHistory_DiffFast, t, 11, NRD_RESOURCE_PERMANENT )
        #if( NRD_MODE == NRD_MODE_SH )
            NRD_INPUT( Texture2D, REBLUR_SH_TYPE, gIn_DiffSh, t, 12, NRD_RESOURCE_IN_DISPATCH )
            NRD_INPUT( Texture2D, REBLUR_SH_TYPE, gHistory_DiffSh, t, 13, NRD_RESOURCE_PERMANENT )
        #endif
    #else
        NRD_INPUT( Texture2D, float, gIn_SpecConfidence, t, 8, NRD_RESOURCE_IN_ZERO_OFFSET )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gIn_Spec, t, 9, NRD_RESOURCE_IN_DISPATCH )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gHistory_Spec, t, 10, NRD_RESOURCE_PERMANENT )
        NRD_INPUT( Texture2D, REBLUR_FAST_TYPE, gHistory_SpecFast, t, 11, NRD_RESOURCE_PERMANENT )
        NRD_INPUT( Texture2D, float, gPrev_SpecHitDistForTracking, t, 12, NRD_RESOURCE_PERMANENT )
        #if( NRD_MODE != NRD_MODE_OCCLUSION )
            NRD_INPUT( Texture2D, float, gIn_SpecHitDistForTracking, t, 13, NRD_RESOURCE_TRANSIENT )
        #endif
        #if( NRD_MODE == NRD_MODE_SH )
            NRD_INPUT( Texture2D, REBLUR_SH_TYPE, gIn_SpecSh, t, 14, NRD_RESOURCE_IN_DISPATCH )
            NRD_INPUT( Texture2D, REBLUR_SH_TYPE, gHistory_SpecSh, t, 15, NRD_RESOURCE_PERMANENT )
        #endif
    #endif
NRD_INPUTS_END

NRD_OUTPUTS_START
    NRD_OUTPUT( RWTexture2D, REBLUR_DATA1_TYPE, gOut_Data1, u, 0, NRD_RESOURCE_TRANSIENT )
    #if( NRD_HAS_DIFF && NRD_HAS_SPEC )
        NRD_OUTPUT( RWTexture2D, REBLUR_TYPE, gOut_Diff, u, 1, NRD_RESOURCE_TRANSIENT )
        NRD_OUTPUT( RWTexture2D, REBLUR_TYPE, gOut_Spec, u, 2, NRD_RESOURCE_TRANSIENT )
        NRD_OUTPUT( RWTexture2D, REBLUR_FAST_TYPE, gOut_DiffFast, u, 3, NRD_RESOURCE_TRANSIENT )
        NRD_OUTPUT( RWTexture2D, REBLUR_FAST_TYPE, gOut_SpecFast, u, 4, NRD_RESOURCE_TRANSIENT )
        NRD_OUTPUT( RWTexture2D, float, gOut_SpecHitDistForTracking, u, 5, NRD_RESOURCE_PERMANENT )
        #if( NRD_MODE != NRD_MODE_OCCLUSION )
            NRD_OUTPUT( RWTexture2D, uint, gOut_Data2, u, 6, NRD_RESOURCE_TRANSIENT )
        #endif
        #if( NRD_MODE == NRD_MODE_SH )
            NRD_OUTPUT( RWTexture2D, REBLUR_SH_TYPE, gOut_DiffSh, u, 7, NRD_RESOURCE_TRANSIENT )
            NRD_OUTPUT( RWTexture2D, REBLUR_SH_TYPE, gOut_SpecSh, u, 8, NRD_RESOURCE_TRANSIENT )
        #endif
    #elif( NRD_HAS_DIFF )
        NRD_OUTPUT( RWTexture2D, REBLUR_TYPE, gOut_Diff, u, 1, NRD_RESOURCE_TRANSIENT )
        NRD_OUTPUT( RWTexture2D, REBLUR_FAST_TYPE, gOut_DiffFast, u, 2, NRD_RESOURCE_TRANSIENT )
        #if( NRD_MODE != NRD_MODE_OCCLUSION )
            NRD_OUTPUT( RWTexture2D, uint, gOut_Data2, u, 3, NRD_RESOURCE_TRANSIENT )
        #endif
        #if( NRD_MODE == NRD_MODE_SH )
            NRD_OUTPUT( RWTexture2D, REBLUR_SH_TYPE, gOut_DiffSh, u, 4, NRD_RESOURCE_TRANSIENT )
        #endif
    #else
        NRD_OUTPUT( RWTexture2D, REBLUR_TYPE, gOut_Spec, u, 1, NRD_RESOURCE_TRANSIENT )
        NRD_OUTPUT( RWTexture2D, REBLUR_FAST_TYPE, gOut_SpecFast, u, 2, NRD_RESOURCE_TRANSIENT )
        NRD_OUTPUT( RWTexture2D, float, gOut_SpecHitDistForTracking, u, 3, NRD_RESOURCE_PERMANENT )
        #if( NRD_MODE != NRD_MODE_OCCLUSION )
            NRD_OUTPUT( RWTexture2D, uint, gOut_Data2, u, 4, NRD_RESOURCE_TRANSIENT )
        #endif
        #if( NRD_MODE == NRD_MODE_SH )
            NRD_OUTPUT( RWTexture2D, REBLUR_SH_TYPE, gOut_SpecSh, u, 5, NRD_RESOURCE_TRANSIENT )
        #endif
    #endif
NRD_OUTPUTS_END

// Macro magic
#define REBLUR_TemporalAccumulationGroupX 8
#define REBLUR_TemporalAccumulationGroupY 16

// Shader only
#ifndef __cplusplus

#define NRD_BORDER 1

#define GROUP_X REBLUR_TemporalAccumulationGroupX
#define GROUP_Y REBLUR_TemporalAccumulationGroupY

#endif
