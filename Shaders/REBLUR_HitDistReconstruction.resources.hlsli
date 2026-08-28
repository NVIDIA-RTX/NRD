/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

NRD_CONSTANTS_START( REBLUR_HitDistReconstructionConstants )
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
    #if( NRD_HAS_DIFF && NRD_HAS_SPEC )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gIn_Diff, t, 3, NRD_RESOURCE_IN )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gIn_Spec, t, 4, NRD_RESOURCE_IN )
    #elif( NRD_HAS_DIFF )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gIn_Diff, t, 3, NRD_RESOURCE_IN )
    #else
        NRD_INPUT( Texture2D, REBLUR_TYPE, gIn_Spec, t, 3, NRD_RESOURCE_IN )
    #endif
NRD_INPUTS_END

NRD_OUTPUTS_START
    #if( NRD_HAS_DIFF && NRD_HAS_SPEC )
        NRD_OUTPUT( RWTexture2D, REBLUR_TYPE, gOut_Diff, u, 0, NRD_RESOURCE_OUT_DISPATCH )
        NRD_OUTPUT( RWTexture2D, REBLUR_TYPE, gOut_Spec, u, 1, NRD_RESOURCE_OUT_DISPATCH )
    #elif( NRD_HAS_DIFF )
        NRD_OUTPUT( RWTexture2D, REBLUR_TYPE, gOut_Diff, u, 0, NRD_RESOURCE_OUT_DISPATCH )
    #else
        NRD_OUTPUT( RWTexture2D, REBLUR_TYPE, gOut_Spec, u, 0, NRD_RESOURCE_OUT_DISPATCH )
    #endif
NRD_OUTPUTS_END

// Macro magic
#define REBLUR_HitDistReconstructionGroupX 8
#define REBLUR_HitDistReconstructionGroupY 16

// Shader only
#ifndef __cplusplus

#if( MODE_5X5 == 1 )
    #define NRD_BORDER 2
#else
    #define NRD_BORDER 1
#endif

#define GROUP_X REBLUR_HitDistReconstructionGroupX
#define GROUP_Y REBLUR_HitDistReconstructionGroupY

#endif
