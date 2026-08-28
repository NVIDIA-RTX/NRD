/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

NRD_CONSTANTS_START( RELAX_PrePassConstants )
    RELAX_SHARED_CONSTANTS
NRD_CONSTANTS_END

NRD_SAMPLERS_START
    NRD_SAMPLER( SamplerState, gNearestClamp, s, 0 )
    NRD_SAMPLER( SamplerState, gLinearClamp, s, 1 )
NRD_SAMPLERS_END

NRD_INPUTS_START
    NRD_INPUT( Texture2D, float, gIn_Tiles, t, 0, NRD_RESOURCE_TRANSIENT )
    NRD_INPUT( Texture2D, float4, gIn_Normal_Roughness, t, 1, NRD_RESOURCE_IN )
    NRD_INPUT( Texture2D, float, gIn_ViewZ, t, 2, NRD_RESOURCE_IN )
    #if( NRD_HAS_DIFF && NRD_HAS_SPEC )
        NRD_INPUT( Texture2D, float4, gIn_Spec, t, 3, NRD_RESOURCE_IN_DISPATCH )
        NRD_INPUT( Texture2D, float4, gIn_Diff, t, 4, NRD_RESOURCE_IN_DISPATCH )
        #if( NRD_MODE == NRD_MODE_SH )
            NRD_INPUT( Texture2D, RELAX_SH_TYPE, gIn_SpecSh, t, 5, NRD_RESOURCE_IN_DISPATCH )
            NRD_INPUT( Texture2D, RELAX_SH_TYPE, gIn_DiffSh, t, 6, NRD_RESOURCE_IN_DISPATCH )
        #endif
    #elif( NRD_HAS_DIFF )
        NRD_INPUT( Texture2D, float4, gIn_Diff, t, 3, NRD_RESOURCE_IN_DISPATCH )
        #if( NRD_MODE == NRD_MODE_SH )
            NRD_INPUT( Texture2D, RELAX_SH_TYPE, gIn_DiffSh, t, 4, NRD_RESOURCE_IN_DISPATCH )
        #endif
    #else
        NRD_INPUT( Texture2D, float4, gIn_Spec, t, 3, NRD_RESOURCE_IN_DISPATCH )
        #if( NRD_MODE == NRD_MODE_SH )
            NRD_INPUT( Texture2D, RELAX_SH_TYPE, gIn_SpecSh, t, 4, NRD_RESOURCE_IN_DISPATCH )
        #endif
    #endif
NRD_INPUTS_END

NRD_OUTPUTS_START
    #if( NRD_HAS_DIFF && NRD_HAS_SPEC )
        NRD_OUTPUT( RWTexture2D, float4, gOut_Spec, u, 0, NRD_RESOURCE_OUT )
        NRD_OUTPUT( RWTexture2D, float4, gOut_Diff, u, 1, NRD_RESOURCE_OUT )
        #if( NRD_MODE == NRD_MODE_SH )
            NRD_OUTPUT( RWTexture2D, RELAX_SH_TYPE, gOut_SpecSh, u, 2, NRD_RESOURCE_OUT )
            NRD_OUTPUT( RWTexture2D, RELAX_SH_TYPE, gOut_DiffSh, u, 3, NRD_RESOURCE_OUT )
        #endif
    #elif( NRD_HAS_DIFF )
        NRD_OUTPUT( RWTexture2D, float4, gOut_Diff, u, 0, NRD_RESOURCE_OUT )
        #if( NRD_MODE == NRD_MODE_SH )
            NRD_OUTPUT( RWTexture2D, RELAX_SH_TYPE, gOut_DiffSh, u, 1, NRD_RESOURCE_OUT )
        #endif
    #else
        NRD_OUTPUT( RWTexture2D, float4, gOut_Spec, u, 0, NRD_RESOURCE_OUT )
        #if( NRD_MODE == NRD_MODE_SH )
            NRD_OUTPUT( RWTexture2D, RELAX_SH_TYPE, gOut_SpecSh, u, 1, NRD_RESOURCE_OUT )
        #endif
    #endif
NRD_OUTPUTS_END

// Macro magic
#define RELAX_PrePassGroupX 16
#define RELAX_PrePassGroupY 16

// Shader only
#ifndef __cplusplus

#define GROUP_X RELAX_PrePassGroupX
#define GROUP_Y RELAX_PrePassGroupY

#endif
