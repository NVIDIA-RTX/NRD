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

#include "SIGMA_Config.hlsli"
#include "SIGMA_Copy.resources.hlsli"

#include "Common.hlsli"

[numthreads( GROUP_X, GROUP_Y, 1 )]
NRD_EXPORT void NRD_CS_MAIN( NRD_CS_MAIN_ARGS )
{
    NRD_CTA_ORDER_DEFAULT;

    if( any( pixelPos >= int2( gRectSizePrev ) ) )
        return;

    // Tile-based early out
    float isSky = NRD_SURFACE( gIn_Tiles, pixelPos >> 4 ).x;
    if( isSky != 0.0 && !gIsRectChanged )
        return;

    // TODO: introduce "CopyResource" in NRD API?
    NRD_SURFACE( gOut_History, pixelPos ) = NRD_SURFACE( gIn_History, pixelPos );
    NRD_SURFACE( gOut_HistoryLength, pixelPos ) = NRD_SURFACE( gIn_HistoryLength, pixelPos );
}
