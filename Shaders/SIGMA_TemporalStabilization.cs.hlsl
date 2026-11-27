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
#include "SIGMA_TemporalStabilization.resources.hlsli"

#include "Common.hlsli"

#include "SIGMA_Common.hlsli"

groupshared float s_Penumbra[ BUFFER_Y ][ BUFFER_X ];
groupshared SIGMA_TYPE s_Shadow_Translucency[ BUFFER_Y ][ BUFFER_X ];

void Preload( uint2 sharedPos, int2 globalPos )
{
    globalPos = clamp( globalPos, 0, gRectSizeMinusOne );

    SIGMA_TYPE s = NRD_SURFACE( gIn_Shadow_Translucency, globalPos );
    s = SIGMA_BackEnd_UnpackShadow( s );

    s_Shadow_Translucency[ sharedPos.y ][ sharedPos.x ] = s;
    s_Penumbra[ sharedPos.y ][ sharedPos.x ] = NRD_SURFACE( gIn_Penumbra, globalPos );
}

uint PackViewZAndHistoryLength( float viewZ, float historyLength )
{
    uint p = asuint( viewZ ) & ~7;
    p |= min( uint( historyLength + 0.5 ), 7 );

    return p;
}

void BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights(
    float2 samplePos, float2 invResourceSize,
    float4 bilinearCustomWeights, bool useBicubic,
    Texture2D<SIGMA_TYPE> tex0, out SIGMA_TYPE c0 )
{
    _BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights_Init;
    _BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights_Color( c0, tex0 );
}

[numthreads( GROUP_X, GROUP_Y, 1 )]
NRD_EXPORT void NRD_CS_MAIN( NRD_CS_MAIN_ARGS )
{
    NRD_CTA_ORDER_DEFAULT;

    // Preload
    float isSky = NRD_SURFACE( gIn_Tiles, pixelPos >> 4 ).x;
    PRELOAD_INTO_SMEM_WITH_TILE_CHECK;

    // Tile-based early out
    if( isSky != 0.0 || any( pixelPos > gRectSizeMinusOne ) )
        return;

    // Center data
    int2 smemPos = threadPos + NRD_BORDER;
    float centerPenumbra = s_Penumbra[ smemPos.y ][ smemPos.x ];
    float viewZ = UnpackViewZ( NRD_SURFACE( gIn_ViewZ, pixelPos ) );

    // Early out #1
    if( !IsInDenoisingRange( viewZ ) )
        return;

    // Early out #2
    float2 pixelUv = float2( pixelPos + 0.5 ) * gRectSizeInv;
    float tileValue = TextureCubic( gIn_Tiles, pixelUv * gResolutionScale ).y;
    bool earlyOut = CanSkipTemporal( tileValue ) || IsBackfaced( centerPenumbra );

    if( earlyOut && SIGMA_SHOW == 0 )
    {
        NRD_SURFACE( gOut_Shadow_Translucency, pixelPos ) = PackShadow( s_Shadow_Translucency[ smemPos.y ][ smemPos.x ] );
        NRD_SURFACE( gOut_HistoryLength, pixelPos ) = PackViewZAndHistoryLength( viewZ, SIGMA_MAX_ACCUM_FRAME_NUM ); // TODO: yes, SIGMA_MAX_ACCUM_FRAME_NUM to allow accumulation in neighbors

        return;
    }

    // Local variance
    float sum = 0.0;
    SIGMA_TYPE m1 = 0;
    SIGMA_TYPE m2 = 0;
    SIGMA_TYPE input = 0;

    [unroll]
    for( j = 0; j <= NRD_BORDER * 2; j++ )
    {
        [unroll]
        for( i = 0; i <= NRD_BORDER * 2; i++ )
        {
            int2 pos = threadPos + int2( i, j );
            SIGMA_TYPE s = s_Shadow_Translucency[ pos.y ][ pos.x ];

            float w = 1.0;
            if( i == NRD_BORDER && j == NRD_BORDER )
                input = s;
            else
            {
                float penum = s_Penumbra[ pos.y ][ pos.x ];

                w = float( IsBackfaced( centerPenumbra ) == IsBackfaced( penum ) );
                w *= GetGaussianWeight( length( float2( i - NRD_BORDER, j - NRD_BORDER ) / NRD_BORDER ) );
            }

            m1 += s * w;
            m2 += s * s * w;
            sum += w;
        }
    }

    m1 /= sum; // sum can't be 0
    m2 /= sum;

    SIGMA_TYPE sigma = GetStdDev( m1, m2 );

    float penumbraRadiusInPixels = centerPenumbra / PixelRadiusToWorld( gUnproject, gOrthoMode, 1.0, viewZ );
    penumbraRadiusInPixels = IsLit( centerPenumbra ) ? 0.0 : penumbraRadiusInPixels;

    // Fade jitter compensation out for subpixel penumbras to preserve sharp shadows
    float jitterScale = saturate( penumbraRadiusInPixels / 0.5 );
    float2 jitter = gJitter * jitterScale;
    float2 jitterPrev = gJitterPrev * jitterScale;

    // Current and previous positions
    float3 Xv = Geometry::ReconstructViewPosition( pixelUv + jitter, gFrustum, viewZ, gOrthoMode );
    float3 X = Geometry::RotateVectorInverse( gWorldToView, Xv );

    float3 mv = NRD_SURFACE( gIn_Mv, pixelPos ) * gMvScale.xyz + gMvBias.xyz;
    float3 Xprev = X;
    float2 smbPixelUv = pixelUv + mv.xy + jitter - jitterPrev;

    if( gMvScale.w == 0.0 )
    {
        if( gMvScale.z == 0.0 )
            mv.z = Geometry::AffineTransform( gWorldToViewPrev, X ).z - viewZ;

        float viewZprev = viewZ + mv.z;
        float3 Xvprevlocal = Geometry::ReconstructViewPosition( smbPixelUv + jitterPrev, gFrustumPrev, viewZprev, gOrthoMode ); // TODO: use gOrthoModePrev

        Xprev = Geometry::RotateVectorInverse( gWorldToViewPrev, Xvprevlocal ) + gCameraDelta.xyz;
    }
    else
    {
        Xprev += mv;
        smbPixelUv = Geometry::GetScreenUv( gWorldToClipPrev, Xprev ) - jitterPrev;
    }

    // History length
    Filtering::Bilinear smbBilinearFilter = Filtering::GetBilinearFilter( smbPixelUv, gRectSizePrev );
    float2 smbBilinearGatherUv = ( NRD_PIXEL_POS( gIn_HistoryLength, smbBilinearFilter.origin ) + 1.0 ) * gResourceSizeInvPrev;
    uint4 prevData = gIn_HistoryLength.GatherRed( gNearestClamp, smbBilinearGatherUv ).wzxy;
    float4 prevViewZ = asfloat( prevData & ~7 );
    float4 prevHistoryLength = float4( prevData & 7 );

    float frustumSize = GetFrustumSize( gMinRectDimMulUnproject, gOrthoMode, viewZ );
    float4 disocclusionThreshold = GetDisocclusionThreshold( NRD_DISOCCLUSION_THRESHOLD, frustumSize, 1.0 ); // TODO: slope scale?
    disocclusionThreshold *= IsInScreenBilinear( smbBilinearFilter.origin, gRectSizePrev );
    disocclusionThreshold -= NRD_EPS;

    float3 Xvprev = Geometry::AffineTransform( gWorldToViewPrev, Xprev );
    float4 smbPlaneDist = abs( prevViewZ - Xvprev.z );
    float4 smbOcclusion = step( smbPlaneDist, disocclusionThreshold );

    float4 smbOcclusionWeights = Filtering::GetBilinearCustomWeights( smbBilinearFilter, smbOcclusion );
    float historyLength = Filtering::ApplyBilinearCustomWeights( prevHistoryLength.x, prevHistoryLength.y, prevHistoryLength.z, prevHistoryLength.w, smbOcclusionWeights );

    // Sample history
    bool isCatRomAllowed = SIGMA_USE_CATROM == 1 && dot( smbOcclusion, 1.0 ) > 3.5 && IsInScreenBicubic( smbBilinearFilter.origin, gRectSizePrev );

    SIGMA_TYPE history;
    BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights(
        saturate( smbPixelUv ) * gRectSizePrev, gResourceSizeInvPrev,
        smbOcclusionWeights, isCatRomAllowed,
        gIn_History, history );

    history = saturate( history );
    history = SIGMA_BackEnd_UnpackShadow( history );

    // Clamp history
    sigma *= lerp( SIGMA_TS_SIGMA_SCALE, 1.0, 1.0 / ( 1.0 + historyLength ) );

    SIGMA_TYPE inputMin = m1 - sigma;
    SIGMA_TYPE inputMax = m1 + sigma;
    SIGMA_TYPE historyClamped = clamp( history, inputMin, inputMax );

    // Antilag
    float antilag = abs( historyClamped.x - history.x );
    antilag = 1.0 - Math::Sqrt01( antilag );
    historyLength *= antilag;

    // Increase history confidence for "in penumbra" motion
    float2 motionInPixels = ( smbPixelUv - pixelUv + jitterPrev - jitter ) * gRectSize;
    float motionInPenumbra = max( length( motionInPixels ), 0.1 ) * Math::PositiveRcp( 0.75 * 2.0 * penumbraRadiusInPixels ); // 75% of size
    float historyConfidence = Math::SmoothStep01( motionInPenumbra );

    // Increase history confidence if clamping didn't occur
    SIGMA_TYPE historyDelta = abs( historyClamped - history );
    float historyDeltaMax = historyDelta.x;
    #if( TRANSLUCENCY == 1 )
        historyDeltaMax = max( max( historyDelta.x, historyDelta.y ), max( historyDelta.z, historyDelta.w ) );
    #endif
    historyConfidence *= saturate( historyDeltaMax * 4.0 );

    // Decorrelate color clamping artefacts probabilistically based on hsitory confidence (0 = confident)
    Rng::Hash::Initialize( pixelPos, gFrameIndex );
    historyClamped = Rng::Hash::GetFloat( ) >= historyConfidence ? history : historyClamped;

    // Accumulation speed
    float accumSpeed = 1.0 / ( 1.0 + historyLength );
    accumSpeed = max( 1.0 - gStabilizationStrength, accumSpeed );

    #if( NRD_SUPPORTS_CHECKERBOARD == 1 )
        if( gCheckerboard != 2 && Sequence::CheckerBoard( pixelPos, gFrameIndex ) != gCheckerboard )
            accumSpeed *= lerp( 1.0 - gCheckerboardResolveAccumSpeed, 1.0, accumSpeed );
    #endif

    // Combine with the current frame
    SIGMA_TYPE result = lerp( historyClamped, input, accumSpeed );

    // Debug ( don't forget that ".x" is used in antilag computations! )
    #if( SIGMA_SHOW == SIGMA_SHOW_TILES )
        tileValue = NRD_SURFACE( gIn_Tiles, pixelPos >> 4 ).y;
        tileValue = !CanSkipSpatial( tileValue ); // optional, just to show fully discarded tiles

        #if( TRANSLUCENCY == 1 )
            result = lerp( float4( 0, 0, 1, 0 ), result, tileValue );
        #else
            result = tileValue;
        #endif

        result *= all( ( pixelPos & 15 ) != 0 );
    #elif( SIGMA_SHOW == SIGMA_SHOW_HISTORY_WEIGHT )
        #if( TRANSLUCENCY == 1 )
            result.yzw = ( 1.0 - accumSpeed ) * float( !earlyOut );
        #endif
    #elif( SIGMA_SHOW == SIGMA_SHOW_HISTORY_LENGTH )
        #if( TRANSLUCENCY == 1 )
            result.yzw = historyLength / SIGMA_MAX_ACCUM_FRAME_NUM;
        #endif
    #elif( SIGMA_SHOW == SIGMA_SHOW_PENUMBRA_SIZE )
        result = centerPenumbra; // like in SplitScreen
    #endif

    // Update history length for the next frame
    historyLength = min( historyLength + 1.0, SIGMA_MAX_ACCUM_FRAME_NUM );

    // Output
    NRD_SURFACE( gOut_Shadow_Translucency, pixelPos ) = PackShadow( result );
    NRD_SURFACE( gOut_HistoryLength, pixelPos ) = PackViewZAndHistoryLength( viewZ, historyLength );
}
