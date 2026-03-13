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
#include "REBLUR_PrePass.resources.hlsli"

#include "Common.hlsli"

#include "REBLUR_Common.hlsli"

[numthreads( GROUP_X, GROUP_Y, 1 )]
NRD_EXPORT void NRD_CS_MAIN( NRD_CS_MAIN_ARGS )
{
    NRD_CTA_ORDER_REVERSED;

    // Tile-based early out
    float isSky = gIn_Tiles[ pixelPos >> 4 ].x;
    if( isSky != 0.0 || any( pixelPos > gRectSizeMinusOne ) )
        return;

    // Early out
    float viewZ = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( pixelPos ) ] );
    if( viewZ > gDenoisingRange )
        return;

    // Center data
    float materialID;
    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ WithRectOrigin( pixelPos ) ], materialID );
    float3 N = normalAndRoughness.xyz;
    float3 Nv = Geometry::RotateVectorInverse( gViewToWorld, N );
    float roughness = normalAndRoughness.w;

    float2 pixelUv = float2( pixelPos + 0.5 ) * gRectSizeInv;
    float3 Xv = Geometry::ReconstructViewPosition( pixelUv, gFrustum, viewZ, gOrthoMode );

    float3 Vv = GetViewVector( Xv, true );
    float NoV = abs( dot( Nv, Vv ) );

    const float frustumSize = GetFrustumSize( gMinRectDimMulUnproject, gOrthoMode, viewZ );
    const float4 rotator = GetBlurKernelRotation( REBLUR_PRE_PASS_ROTATOR_MODE, pixelPos, gRotatorPre, gFrameIndex );

    // Checkerboard resolve
#if( NRD_SUPPORTS_CHECKERBOARD == 1 )
    uint checkerboard = Sequence::CheckerBoard( pixelPos, gFrameIndex );

    int3 checkerboardPos = pixelPos.xxy + int3( -1, 1, 0 );
    checkerboardPos.x = max( checkerboardPos.x, 0 );
    checkerboardPos.y = min( checkerboardPos.y, gRectSizeMinusOne.x );
    float viewZ0 = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( checkerboardPos.xz ) ] );
    float viewZ1 = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( checkerboardPos.yz ) ] );
    float disocclusionThresholdCheckerboard = GetDisocclusionThreshold( NRD_DISOCCLUSION_THRESHOLD, frustumSize, NoV );
    float2 wc = GetDisocclusionWeight( float2( viewZ0, viewZ1 ), viewZ, disocclusionThresholdCheckerboard );
    wc.x = ( viewZ0 > gDenoisingRange || pixelPos.x < 1 ) ? 0.0 : wc.x;
    wc.y = ( viewZ1 > gDenoisingRange || pixelPos.x >= gRectSizeMinusOne.x ) ? 0.0 : wc.y;
    wc *= Math::PositiveRcp( wc.x + wc.y );
    checkerboardPos.xy >>= 1;
#endif

    // Non-linear accum speed
    float2 nonLinearAccumSpeed = REBLUR_PRE_PASS_NON_LINEAR_ACCUM_SPEED;

    // Spatial filtering
    #define REBLUR_SPATIAL_PASS REBLUR_PRE_PASS

    #if( NRD_DIFF )
        #define REBLUR_SPATIAL_LOBE REBLUR_DIFF
        #define MAX_BLUR_RADIUS gDiffPrepassBlurRadius
        #include "REBLUR_Common_SpatialFilter.hlsli"
    #endif

    #if( NRD_SPEC )
        #define REBLUR_SPATIAL_LOBE REBLUR_SPEC
        #define MAX_BLUR_RADIUS gSpecPrepassBlurRadius
        #include "REBLUR_Common_SpatialFilter.hlsli"
    #endif
}
