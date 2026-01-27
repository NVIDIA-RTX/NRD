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
#include "REBLUR_HistoryFix.resources.hlsli"

#include "Common.hlsli"

#include "REBLUR_Common.hlsli"

groupshared float s_DiffLuma[ BUFFER_Y ][ BUFFER_X ];
groupshared float s_SpecLuma[ BUFFER_Y ][ BUFFER_X ];
groupshared float2 s_FrameNum[ BUFFER_Y ][ BUFFER_X ];

void Preload( uint2 sharedPos, int2 globalPos )
{
    globalPos = clamp( globalPos, 0, gRectSizeMinusOne );

    #if( NRD_DIFF )
        s_DiffLuma[ sharedPos.y ][ sharedPos.x ] = gIn_DiffFast[ globalPos ];
    #endif

    #if( NRD_SPEC )
        s_SpecLuma[ sharedPos.y ][ sharedPos.x ] = gIn_SpecFast[ globalPos ];
    #endif

    // TODO: this is needed only for 1-pixel border, but adding a conditional will break texture batching. Only a minor/zero gain expected...
    s_FrameNum[ sharedPos.y ][ sharedPos.x ] = UnpackData1( gIn_Data1[ globalPos ] );
}

// Tests 20, 23, 24, 27, 28, 54, 59, 65, 66, 76, 81, 98, 112, 117, 124, 126, 128, 134
// TODO: potentially do color clamping after reconstruction in a separate pass

[numthreads( GROUP_X, GROUP_Y, 1 )]
NRD_EXPORT void NRD_CS_MAIN( NRD_CS_MAIN_ARGS )
{
    NRD_CTA_ORDER_REVERSED;

    // Preload
    float isSky = gIn_Tiles[ pixelPos >> 4 ].x;
    PRELOAD_INTO_SMEM_WITH_TILE_CHECK;

    // Tile-based early out
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
    float roughness = normalAndRoughness.w;

    float frustumSize = GetFrustumSize( gMinRectDimMulUnproject, gOrthoMode, viewZ );
    float2 pixelUv = float2( pixelPos + 0.5 ) * gRectSizeInv;
    float3 Xv = Geometry::ReconstructViewPosition( pixelUv, gFrustum, viewZ, gOrthoMode );
    float3 Nv = Geometry::RotateVectorInverse( gViewToWorld, N );
    float3 Vv = GetViewVector( Xv, true );
    float NoV = abs( dot( Nv, Vv ) );

    int2 smemPos = threadPos + NRD_BORDER;
    float2 frameNum = s_FrameNum[ smemPos.y ][ smemPos.x ];

    float invHistoryFixFrameNum = 1.0 / max( gHistoryFixFrameNum, NRD_EPS );
    float2 frameNumAvgNorm = saturate( frameNum * invHistoryFixFrameNum );

    // Use smaller strides if neighborhood pixels have longer history to minimize chances of ruining contact details
    float baseStride = materialID == gHistoryFixAlternatePixelStrideMaterialID ? gHistoryFixAlternatePixelStride : gHistoryFixBasePixelStride;
    baseStride /= 1.0 + 1.0; // to match RELAX, where "frameNum" after "TemporalAccumulation" is "1", not "0"

    float2 stride = 1.0;

    [unroll]
    for( i = -1; i <= 1; i++ )
    {
        [unroll]
        for( j = -1; j <= 1; j++ )
        {
            if( i == 0 && j == 0 )
                continue;

            float2 f = s_FrameNum[ smemPos.y + j ][ smemPos.x + i ];

            stride -= saturate( ( f - frameNum ) * invHistoryFixFrameNum ) / 9.0;
        }
    }

    stride = lerp( 1.0, baseStride, stride );
    stride *= 2.0 / REBLUR_HISTORY_FIX_FILTER_RADIUS; // preserve blur radius in pixels ( default blur radius is 2 taps )
    stride *= float2( frameNum < gHistoryFixFrameNum );

    // Diffuse
    #if( NRD_DIFF )
    {
        REBLUR_TYPE diff = gIn_Diff[ pixelPos ];
        #if( NRD_MODE == SH )
            float4 diffSh = gIn_DiffSh[ pixelPos ];
        #endif

        float diffNonLinearAccumSpeed = 1.0 / ( 1.0 + frameNum.x );

        float hitDistScale = _REBLUR_GetHitDistanceNormalization( viewZ, gHitDistSettings, 1.0 );
        float hitDist = ExtractHitDist( diff ) * hitDistScale;
        float hitDistFactor = GetHitDistFactor( hitDist, frustumSize );
        hitDist = ExtractHitDist( diff );

        // Stride between taps
        float diffStride = stride.x;
        diffStride *= lerp( 0.25 + 0.75 * Math::Sqrt01( hitDistFactor ), 1.0, diffNonLinearAccumSpeed ); // "hitDistFactor" is very noisy and breaks nice patterns
        diffStride = round( diffStride );

        // History reconstruction
        if( diffStride != 0.0 )
        {
            // Parameters
            float normalWeightParam = GetNormalWeightParam( diffNonLinearAccumSpeed, gLobeAngleFraction );
            float2 geometryWeightParams = GetGeometryWeightParams( gPlaneDistSensitivity, frustumSize, Xv, Nv );
            float hitDistWeightNorm = 1.0 / ( 0.5 * diffNonLinearAccumSpeed );

            float sumd = 1.0 + frameNum.x;
            #if( REBLUR_PERFORMANCE_MODE == 1 )
                sumd = 1.0 + 1.0 / ( 1.0 + gMaxAccumulatedFrameNum ) - diffNonLinearAccumSpeed;
            #endif

            diff *= sumd;
            #if( NRD_MODE == SH )
                diffSh *= sumd;
            #endif

            [unroll]
            for( j = -REBLUR_HISTORY_FIX_FILTER_RADIUS; j <= REBLUR_HISTORY_FIX_FILTER_RADIUS; j++ )
            {
                [unroll]
                for( i = -REBLUR_HISTORY_FIX_FILTER_RADIUS; i <= REBLUR_HISTORY_FIX_FILTER_RADIUS; i++ )
                {
                    // Skip center
                    if( i == 0 && j == 0 )
                        continue;

                    // Skip corners
                    if( abs( i ) + abs( j ) == REBLUR_HISTORY_FIX_FILTER_RADIUS * 2 )
                        continue;

                    // Sample uv ( at the pixel center )
                    float2 uv = pixelUv + float2( i, j ) * diffStride * gRectSizeInv;

                    // Apply "mirror" to not waste taps going outside of the screen
                    uv = MirrorUv( uv );

                    // "uv" to "pos"
                    int2 pos = uv * gRectSize; // "uv" can't be "1"

                    // Fetch data
                    float zs = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( pos ) ] );
                    float3 Xvs = Geometry::ReconstructViewPosition( uv, gFrustum, zs, gOrthoMode );

                    float materialIDs;
                    float4 Ns = gIn_Normal_Roughness[ WithRectOrigin( pos ) ];
                    Ns = NRD_FrontEnd_UnpackNormalAndRoughness( Ns, materialIDs );

                    // Weight
                    float angle = Math::AcosApprox( dot( Ns.xyz, N ) );
                    float NoX = dot( Nv, Xvs );

                    float w = ComputeWeight( NoX, geometryWeightParams.x, geometryWeightParams.y );
                    w *= CompareMaterials( materialID, materialIDs, gDiffMinMaterial );
                    w *= ComputeExponentialWeight( angle, normalWeightParam, 0.0 );
                    w = zs < gDenoisingRange ? w : 0.0; // |NoX| can be ~0 if "zs" is out of range
                    // gaussian weight is not needed

                    #if( REBLUR_PERFORMANCE_MODE == 0 )
                        w *= 1.0 + UnpackData1( gIn_Data1[ pos ] ).x;
                    #endif

                    REBLUR_TYPE s = gIn_Diff[ pos ];
                    s = Denanify( w, s );

                    // A-trous weight
                    float hs = ExtractHitDist( s );
                    float d = hs - hitDist; // use normalized hit distanced for simplicity ( no difference )
                    w *= exp( -d * d * hitDistWeightNorm );

                    // Accumulate
                    sumd += w;

                    diff += s * w;
                    #if( NRD_MODE == SH )
                        float4 sh = gIn_DiffSh[ pos ];
                        sh = Denanify( w, sh );
                        diffSh += sh * w;
                    #endif
                }
            }

            sumd = Math::PositiveRcp( sumd );
            diff *= sumd;
            #if( NRD_MODE == SH )
                diffSh *= sumd;
            #endif
        }

        // Local variance
        float diffCenter = 0;
        float diffM1 = 0;
        float diffM2 = 0;
        float diffSpecialM1 = 0;
        float diffSpecialM2 = 0;

        [unroll]
        for( j = -NRD_BORDER; j <= NRD_BORDER; j++ )
        {
            [unroll]
            for( i = -NRD_BORDER; i <= NRD_BORDER; i++ )
            {
                int2 pos = smemPos + int2( i, j );
                float d = s_DiffLuma[ pos.y ][ pos.x ];

                // Center
                if( i == 0 && j == 0 )
                    diffCenter = d;

                // Variance in 5x5 for fast history
                if( abs( i ) <= REBLUR_FAST_HISTORY_CLAMPING_RADIUS && abs( j ) <= REBLUR_FAST_HISTORY_CLAMPING_RADIUS )
                {
                    diffM1 += d;
                    diffM2 += d * d;
                }

                // Variance in "NRD_BORDER x NRD_BORDER" skipping central 3x3 for anti-firefly
                if( NRD_SUPPORTS_ANTIFIREFLY && !( abs( i ) <= 1 && abs( j ) <= 1 ) )
                {
                    diffSpecialM1 += d;
                    diffSpecialM2 += d * d;
                }
            }
        }

        // Anti-firefly
        float diffLuma = GetLuma( diff );

        if( NRD_SUPPORTS_ANTIFIREFLY && gAntiFirefly )
        {
            float invNorm = 1.0 / ( ( NRD_BORDER * 2 + 1 ) * ( NRD_BORDER * 2 + 1 ) - 3 * 3 ); // -9 samples
            diffSpecialM1 *= invNorm;
            diffSpecialM2 *= invNorm;

            float diffSigma = GetStdDev( diffSpecialM1, diffSpecialM2 ) * REBLUR_ANTI_FIREFLY_SIGMA_SCALE;

            diffLuma = clamp( diffLuma, diffSpecialM1 - diffSigma, diffSpecialM1 + diffSigma );
        }

        // Fix fast history
        float f = frameNumAvgNorm.x;
        diffCenter = lerp( diffLuma, diffCenter, f );
        gOut_DiffFast[ pixelPos ] = diffCenter;

        // Clamp to fast history
        {
            float invNorm = 1.0 / ( ( REBLUR_FAST_HISTORY_CLAMPING_RADIUS * 2 + 1 ) * ( REBLUR_FAST_HISTORY_CLAMPING_RADIUS * 2 + 1 ) );
            diffM1 *= invNorm;
            diffM2 *= invNorm;

            float diffSigma = GetStdDev( diffM1, diffM2 ) * gFastHistoryClampingSigmaScale;
            float diffLumaClamped = clamp( diffLuma, diffM1 - diffSigma, diffM1 + diffSigma );

            diffLuma = lerp( diffLumaClamped, diffLuma, 1.0 / ( 1.0 + float( gMaxFastAccumulatedFrameNum < gMaxAccumulatedFrameNum ) * frameNum.x * 2.0 ) );
        }

        // Change luma
        #if( REBLUR_SHOW == REBLUR_SHOW_FAST_HISTORY )
            diffLuma = diffCenter;
        #endif

        diff = ChangeLuma( diff, diffLuma );
        #if( NRD_MODE == SH )
            diffSh.xyz *= GetLumaScale( length( diffSh.xyz ), diffLuma );
        #endif

        // Output
        gOut_Diff[ pixelPos ] = diff;
        #if( NRD_MODE == SH )
            gOut_DiffSh[ pixelPos ] = diffSh;
        #endif
    }
    #endif

    // Specular
    #if( NRD_SPEC )
    {
        REBLUR_TYPE spec = gIn_Spec[ pixelPos ];
        #if( NRD_MODE == SH )
            float4 specSh = gIn_SpecSh[ pixelPos ];
        #endif

        float smc = GetSpecMagicCurve( roughness );
        float specNonLinearAccumSpeed = 1.0 / ( 1.0 + frameNum.y );

        float hitDistScale = _REBLUR_GetHitDistanceNormalization( viewZ, gHitDistSettings, roughness );
        float hitDist = ExtractHitDist( spec ) * hitDistScale;
        #if( NRD_MODE != OCCLUSION )
            // "gIn_SpecHitDistForTracking" is better for low roughness, but doesn't suit for high roughness ( because it's min )
            hitDist = lerp( gIn_SpecHitDistForTracking[ pixelPos ], hitDist, smc );
        #endif
        float hitDistFactor = GetHitDistFactor( hitDist, frustumSize );
        hitDist = saturate( hitDist / hitDistScale );

        // Stride between taps
        float specStride = stride.y;
        specStride *= lerp( 0.25 + 0.75 * Math::Sqrt01( hitDistFactor ), 1.0, specNonLinearAccumSpeed ); // "hitDistFactor" is very noisy and breaks nice patterns
        specStride *= lerp( 0.25, 1.0, smc ); // hand tuned // TODO: use "lobeRadius"?
        specStride = round( specStride );

        // History reconstruction
        if( specStride != 0 )
        {
            // Parameters
            float normalWeightParam = GetNormalWeightParam( specNonLinearAccumSpeed, gLobeAngleFraction, roughness );
            float2 geometryWeightParams = GetGeometryWeightParams( gPlaneDistSensitivity, frustumSize, Xv, Nv );
            float hitDistWeightNorm = 1.0 / ( lerp( 0.005, 0.5, smc ) * specNonLinearAccumSpeed );
            float2 relaxedRoughnessWeightParams = GetRelaxedRoughnessWeightParams( roughness * roughness, sqrt( gRoughnessFraction ) );

            float sums = 1.0 + frameNum.y;
            #if( REBLUR_PERFORMANCE_MODE == 1 )
                sums = 1.0 + 1.0 / ( 1.0 + gMaxAccumulatedFrameNum ) - specNonLinearAccumSpeed;
            #endif

            spec *= sums;
            #if( NRD_MODE == SH )
                specSh.xyz *= sums;
            #endif

            [unroll]
            for( j = -REBLUR_HISTORY_FIX_FILTER_RADIUS; j <= REBLUR_HISTORY_FIX_FILTER_RADIUS; j++ )
            {
                [unroll]
                for( i = -REBLUR_HISTORY_FIX_FILTER_RADIUS; i <= REBLUR_HISTORY_FIX_FILTER_RADIUS; i++ )
                {
                    // Skip center
                    if( i == 0 && j == 0 )
                        continue;

                    // Skip corners
                    if( abs( i ) + abs( j ) == REBLUR_HISTORY_FIX_FILTER_RADIUS * 2 )
                        continue;

                    // Sample uv ( at the pixel center )
                    float2 uv = pixelUv + float2( i, j ) * specStride * gRectSizeInv;

                    // Apply "mirror" to not waste taps going outside of the screen
                    uv = MirrorUv( uv );

                    // "uv" to "pos"
                    int2 pos = uv * gRectSize; // "uv" can't be "1"

                    // Fetch data
                    float zs = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( pos ) ] );
                    float3 Xvs = Geometry::ReconstructViewPosition( uv, gFrustum, zs, gOrthoMode );

                    float materialIDs;
                    float4 Ns = gIn_Normal_Roughness[ WithRectOrigin( pos ) ];
                    Ns = NRD_FrontEnd_UnpackNormalAndRoughness( Ns, materialIDs );

                    // Weight
                    float angle = Math::AcosApprox( dot( Ns.xyz, N ) );
                    float NoX = dot( Nv, Xvs );

                    float w = ComputeWeight( NoX, geometryWeightParams.x, geometryWeightParams.y );
                    w *= CompareMaterials( materialID, materialIDs, gSpecMinMaterial );
                    w *= ComputeExponentialWeight( angle, normalWeightParam, 0.0 );
                    w *= ComputeExponentialWeight( Ns.w * Ns.w, relaxedRoughnessWeightParams.x, relaxedRoughnessWeightParams.y );
                    w = zs < gDenoisingRange ? w : 0.0; // |NoX| can be ~0 if "zs" is out of range
                    // gaussian weight is not needed

                    #if( REBLUR_PERFORMANCE_MODE == 0 )
                        w *= 1.0 + UnpackData1( gIn_Data1[ pos ] ).y;
                    #endif

                    REBLUR_TYPE s = gIn_Spec[ pos ];
                    s = Denanify( w, s );

                    // A-trous weight
                    float hs = ExtractHitDist( s );
                    float d = hs - hitDist; // use normalized hit distances for simplicity ( no difference, roughness weight handles the rest )
                    w *= exp( -d * d * hitDistWeightNorm );

                    // Accumulate
                    sums += w;

                    spec += s * w;
                    #if( NRD_MODE == SH )
                        float4 sh = gIn_SpecSh[ pos ];
                        sh = Denanify( w, sh );
                        specSh.xyz += sh.xyz * w;
                    #endif
                }
            }

            sums = Math::PositiveRcp( sums );
            spec *= sums;
            #if( NRD_MODE == SH )
                specSh.xyz *= sums;
            #endif
        }

        // Local variance
        float specCenter = 0;
        float specM1 = 0;
        float specM2 = 0;
        float specSpecialM1 = 0;
        float specSpecialM2 = 0;

        [unroll]
        for( j = -NRD_BORDER; j <= NRD_BORDER; j++ )
        {
            [unroll]
            for( i = -NRD_BORDER; i <= NRD_BORDER; i++ )
            {
                int2 pos = smemPos + int2( i, j );
                float s = s_SpecLuma[ pos.y ][ pos.x ];

                // Center
                if( i == 0 && j == 0 )
                    specCenter = s;

                // Variance in 5x5 for fast history
                if( abs( i ) <= 2 && abs( j ) <= 2 )
                {
                    specM1 += s;
                    specM2 += s * s;
                }

                // Variance in "NRD_BORDER x NRD_BORDER" skipping central 3x3 for anti-firefly
                if( NRD_SUPPORTS_ANTIFIREFLY && !( abs( i ) <= 1 && abs( j ) <= 1 ) )
                {
                    specSpecialM1 += s;
                    specSpecialM2 += s * s;
                }
            }
        }

        // Anti-firefly
        float specLuma = GetLuma( spec );

        if( NRD_SUPPORTS_ANTIFIREFLY && gAntiFirefly )
        {
            float invNorm = 1.0 / ( ( NRD_BORDER * 2 + 1 ) * ( NRD_BORDER * 2 + 1 ) - 3 * 3 ); // -9 samples
            specSpecialM1 *= invNorm;
            specSpecialM2 *= invNorm;

            float specSigma = GetStdDev( specSpecialM1, specSpecialM2 ) * REBLUR_ANTI_FIREFLY_SIGMA_SCALE;

            specLuma = clamp( specLuma, specSpecialM1 - specSigma, specSpecialM1 + specSigma );
        }

        // Fix fast history
        float f = frameNumAvgNorm.y;
        f = lerp( 1.0, f, smc ); // HistoryFix-ed data is undesired in fast history for low roughness ( test 115 )
        specCenter = lerp( specLuma, specCenter, f );

        gOut_SpecFast[ pixelPos ] = specCenter;

        // Clamp to fast history
        {
            float invNorm = 1.0 / ( ( REBLUR_FAST_HISTORY_CLAMPING_RADIUS * 2 + 1 ) * ( REBLUR_FAST_HISTORY_CLAMPING_RADIUS * 2 + 1 ) );
            specM1 *= invNorm;
            specM2 *= invNorm;

            float fastHistoryClampingSigmaScale = gFastHistoryClampingSigmaScale;
            if( materialID == gStrandMaterialID )
                fastHistoryClampingSigmaScale = max( fastHistoryClampingSigmaScale, 3.0 );

            float specSigma = GetStdDev( specM1, specM2 ) * fastHistoryClampingSigmaScale;
            float specLumaClamped = clamp( specLuma, specM1 - specSigma, specM1 + specSigma );

            specLuma = lerp( specLumaClamped, specLuma, 1.0 / ( 1.0 + float( gMaxFastAccumulatedFrameNum < gMaxAccumulatedFrameNum ) * frameNum.y * 2.0 ) );
        }

        // Change luma
        #if( REBLUR_SHOW == REBLUR_SHOW_FAST_HISTORY )
            specLuma = specCenter;
        #endif

        spec = ChangeLuma( spec, specLuma );
        #if( NRD_MODE == SH )
            specSh.xyz *= GetLumaScale( length( specSh.xyz ), specLuma );
        #endif

        // Output
        gOut_Spec[ pixelPos ] = spec;
        #if( NRD_MODE == SH )
            gOut_SpecSh[ pixelPos ] = specSh;
        #endif
    }
    #endif
}
