/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

[numthreads( GROUP_X, GROUP_Y, 1 )]
NRD_EXPORT void NRD_CS_MAIN( NRD_CS_MAIN_ARGS )
{
    NRD_CTA_ORDER_REVERSED;

    // Tile-based early out
    float isSky = gIn_Tiles[ pixelPos >> 4 ];
    if( isSky != 0.0 || any( pixelPos > gRectSizeMinusOne ) )
        return; // IMPORTANT: no data output, must be rejected by the "viewZ" check!

    // Early out
    float viewZ = UnpackViewZ( gIn_ViewZ[ pixelPos ] );
    if( viewZ > gDenoisingRange )
        return; // IMPORTANT: no data output, must be rejected by the "viewZ" check!

    // Normal and roughness
    float materialID;
    float4 normalAndRoughnessPacked = gIn_Normal_Roughness[ WithRectOrigin( pixelPos ) ];
    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness( normalAndRoughnessPacked, materialID );
    float3 N = normalAndRoughness.xyz;
    float3 Nv = Geometry::RotateVectorInverse( gViewToWorld, N );
    float roughness = normalAndRoughness.w;

    // Shared data
    REBLUR_DATA1_TYPE data1 = UnpackData1( gIn_Data1[ pixelPos ] );

    float2 pixelUv = float2( pixelPos + 0.5 ) * gRectSizeInv;
    float3 Xv = Geometry::ReconstructViewPosition( pixelUv, gFrustum, viewZ, gOrthoMode );

    float3 Vv = GetViewVector( Xv, true );
    float NoV = abs( dot( Nv, Vv ) );

    const float frustumSize = GetFrustumSize( gMinRectDimMulUnproject, gOrthoMode, viewZ );
    const float4 rotator = GetBlurKernelRotation( REBLUR_POST_BLUR_ROTATOR_MODE, pixelPos, gRotatorPost, gFrameIndex );

    // Output
    gOut_Normal_Roughness[ pixelPos ] = normalAndRoughnessPacked;
    #ifdef REBLUR_NO_TEMPORAL_STABILIZATION
        gOut_InternalData[ pixelPos ] = PackInternalData( data1.x + 1.0, data1.y + 1.0, materialID ); // increment history length
    #endif

    // Spatial filtering
    #define REBLUR_SPATIAL_MODE REBLUR_POST_BLUR

    #ifdef REBLUR_DIFFUSE
    {
        float sum = 1.0;
        REBLUR_TYPE diff = gIn_Diff[ pixelPos ];
        #ifdef REBLUR_SH
            float4 diffSh = gIn_DiffSh[ pixelPos ];
        #endif

        #include "REBLUR_Common_DiffuseSpatialFilter.hlsli"
    }
    #endif

    #ifdef REBLUR_SPECULAR
    {
        float sum = 1.0;
        REBLUR_TYPE spec = gIn_Spec[ pixelPos ];
        #ifdef REBLUR_SH
            float4 specSh = gIn_SpecSh[ pixelPos ];
        #endif

        #include "REBLUR_Common_SpecularSpatialFilter.hlsli"
    }
    #endif
}
