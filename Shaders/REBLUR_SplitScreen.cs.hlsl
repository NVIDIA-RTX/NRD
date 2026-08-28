/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "NRD.hlsli"
#include "ml.hlsli"

#include "REBLUR_Config.hlsli"
#include "REBLUR_SplitScreen.resources.hlsli"

#include "Common.hlsli"

#include "REBLUR_Common.hlsli"

[numthreads( GROUP_X, GROUP_Y, 1 )]
NRD_EXPORT void NRD_CS_MAIN( NRD_CS_MAIN_ARGS )
{
    NRD_CTA_ORDER_DEFAULT;

    float2 pixelUv = float2( pixelPos + 0.5 ) * gRectSizeInv;
    if( pixelUv.x > gSplitScreen || any( pixelPos > gRectSizeMinusOne ) )
        return;

    float viewZ = UnpackViewZ( NRD_SURFACE( gIn_ViewZ, pixelPos ) );
    uint2 checkerboardPos = pixelPos;

    #if( NRD_HAS_DIFF )
        checkerboardPos.x = pixelPos.x >> ( gDiffCheckerboard != 2 ? 1 : 0 );

        REBLUR_TYPE diff = NRD_SURFACE( gIn_Diff, checkerboardPos );
        NRD_SURFACE( gOut_Diff, pixelPos ) = diff * float( IsInDenoisingRange( viewZ ) );

        #if( NRD_MODE == NRD_MODE_SH )
            REBLUR_SH_TYPE diffSh = NRD_SURFACE( gIn_DiffSh, checkerboardPos );
            NRD_SURFACE( gOut_DiffSh, pixelPos ) = diffSh * float( IsInDenoisingRange( viewZ ) );
        #endif
    #endif

    #if( NRD_HAS_SPEC )
        checkerboardPos.x = pixelPos.x >> ( gSpecCheckerboard != 2 ? 1 : 0 );

        REBLUR_TYPE spec = NRD_SURFACE( gIn_Spec, checkerboardPos );
        NRD_SURFACE( gOut_Spec, pixelPos ) = spec * float( IsInDenoisingRange( viewZ ) );

        #if( NRD_MODE == NRD_MODE_SH )
            REBLUR_SH_TYPE specSh = NRD_SURFACE( gIn_SpecSh, checkerboardPos );
            NRD_SURFACE( gOut_SpecSh, pixelPos ) = specSh * float( IsInDenoisingRange( viewZ ) );
        #endif
    #endif
}
