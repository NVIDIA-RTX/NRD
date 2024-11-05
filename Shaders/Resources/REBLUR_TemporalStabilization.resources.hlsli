/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

NRD_CONSTANTS_START( REBLUR_TemporalStabilizationConstants )
    REBLUR_SHARED_CONSTANTS
NRD_CONSTANTS_END

NRD_SAMPLERS_START
    NRD_SAMPLER( SamplerState, gNearestClamp, s, 0 )
    NRD_SAMPLER( SamplerState, gLinearClamp, s, 1 )
NRD_SAMPLERS_END

#if( defined REBLUR_DIFFUSE && defined REBLUR_SPECULAR )

    NRD_INPUTS_START
        NRD_INPUT( Texture2D<float>, gIn_Tiles, t, 0 )
        NRD_INPUT( Texture2D<float4>, gIn_Normal_Roughness, t, 1 )
        NRD_INPUT( Texture2D<float4>, gIn_BaseColor_Metalness, t, 2 )
        NRD_INPUT( Texture2D<float>, gIn_ViewZ, t, 3 )
        NRD_INPUT( Texture2D<REBLUR_DATA1_TYPE>, gIn_Data1, t, 4 )
        NRD_INPUT( Texture2D<uint>, gIn_Data2, t, 5 )
        NRD_INPUT( Texture2D<REBLUR_TYPE>, gIn_Diff, t, 6 )
        NRD_INPUT( Texture2D<REBLUR_TYPE>, gIn_Spec, t, 7 )
        NRD_INPUT( Texture2D<REBLUR_TYPE>, gHistory_DiffStabilized, t, 8 )
        NRD_INPUT( Texture2D<REBLUR_TYPE>, gHistory_SpecStabilized, t, 9 )
        NRD_INPUT( Texture2D<float>, gIn_SpecHitDistForTracking, t, 10 )
        #ifdef REBLUR_SH
            NRD_INPUT( Texture2D<REBLUR_SH_TYPE>, gIn_DiffSh, t, 11 )
            NRD_INPUT( Texture2D<REBLUR_SH_TYPE>, gIn_SpecSh, t, 12 )
            NRD_INPUT( Texture2D<REBLUR_SH_TYPE>, gHistory_DiffShStabilized, t, 13 )
            NRD_INPUT( Texture2D<REBLUR_SH_TYPE>, gHistory_SpecShStabilized, t, 14 )
        #endif
    NRD_INPUTS_END

    NRD_OUTPUTS_START
        NRD_OUTPUT( RWTexture2D<float4>, gInOut_Mv, u, 0 )
        NRD_OUTPUT( RWTexture2D<uint>, gOut_InternalData, u, 1 )
        NRD_OUTPUT( RWTexture2D<REBLUR_TYPE>, gOut_Diff, u, 2 )
        NRD_OUTPUT( RWTexture2D<REBLUR_TYPE>, gOut_Spec, u, 3 )
        #ifdef REBLUR_SH
            NRD_OUTPUT( RWTexture2D<REBLUR_SH_TYPE>, gOut_DiffSh, u, 4 )
            NRD_OUTPUT( RWTexture2D<REBLUR_SH_TYPE>, gOut_SpecSh, u, 5 )
        #endif
    NRD_OUTPUTS_END

#elif( defined REBLUR_DIFFUSE )

    NRD_INPUTS_START
        NRD_INPUT( Texture2D<float>, gIn_Tiles, t, 0 )
        NRD_INPUT( Texture2D<float4>, gIn_Normal_Roughness, t, 1 )
        NRD_INPUT( Texture2D<float>, gIn_ViewZ, t, 2 )
        NRD_INPUT( Texture2D<REBLUR_DATA1_TYPE>, gIn_Data1, t, 3 )
        NRD_INPUT( Texture2D<uint>, gIn_Data2, t, 4 )
        NRD_INPUT( Texture2D<REBLUR_TYPE>, gIn_Diff, t, 5 )
        NRD_INPUT( Texture2D<REBLUR_TYPE>, gHistory_DiffStabilized, t, 6 )
        #ifdef REBLUR_SH
            NRD_INPUT( Texture2D<REBLUR_SH_TYPE>, gIn_DiffSh, t, 7 )
            NRD_INPUT( Texture2D<REBLUR_SH_TYPE>, gHistory_DiffShStabilized, t, 8 )
        #endif
    NRD_INPUTS_END

    NRD_OUTPUTS_START
        NRD_OUTPUT( RWTexture2D<float4>, gInOut_Mv, u, 0 )
        NRD_OUTPUT( RWTexture2D<uint>, gOut_InternalData, u, 1 )
        NRD_OUTPUT( RWTexture2D<REBLUR_TYPE>, gOut_Diff, u, 2 )
        #ifdef REBLUR_SH
            NRD_OUTPUT( RWTexture2D<REBLUR_SH_TYPE>, gOut_DiffSh, u, 3 )
        #endif
    NRD_OUTPUTS_END

#else

    NRD_INPUTS_START
        NRD_INPUT( Texture2D<float>, gIn_Tiles, t, 0 )
        NRD_INPUT( Texture2D<float4>, gIn_Normal_Roughness, t, 1 )
        NRD_INPUT( Texture2D<float4>, gIn_BaseColor_Metalness, t, 2 )
        NRD_INPUT( Texture2D<float>, gIn_ViewZ, t, 3 )
        NRD_INPUT( Texture2D<REBLUR_DATA1_TYPE>, gIn_Data1, t, 4 )
        NRD_INPUT( Texture2D<uint>, gIn_Data2, t, 5 )
        NRD_INPUT( Texture2D<REBLUR_TYPE>, gIn_Spec, t, 6 )
        NRD_INPUT( Texture2D<REBLUR_TYPE>, gHistory_SpecStabilized, t, 7 )
        NRD_INPUT( Texture2D<float>, gIn_SpecHitDistForTracking, t, 8 )
        #ifdef REBLUR_SH
            NRD_INPUT( Texture2D<REBLUR_SH_TYPE>, gIn_SpecSh, t, 9 )
            NRD_INPUT( Texture2D<REBLUR_SH_TYPE>, gHistory_SpecShStabilized, t, 10 )
        #endif
    NRD_INPUTS_END

    NRD_OUTPUTS_START
        NRD_OUTPUT( RWTexture2D<float4>, gInOut_Mv, u, 0 )
        NRD_OUTPUT( RWTexture2D<uint>, gOut_InternalData, u, 1 )
        NRD_OUTPUT( RWTexture2D<REBLUR_TYPE>, gOut_Spec, u, 2 )
        #ifdef REBLUR_SH
            NRD_OUTPUT( RWTexture2D<REBLUR_SH_TYPE>, gOut_SpecSh, u, 3 )
        #endif
    NRD_OUTPUTS_END

#endif

// Macro magic
#define REBLUR_TemporalStabilizationGroupX 8
#define REBLUR_TemporalStabilizationGroupY 16

// Redirection
#undef GROUP_X
#undef GROUP_Y
#define GROUP_X REBLUR_TemporalStabilizationGroupX
#define GROUP_Y REBLUR_TemporalStabilizationGroupY
