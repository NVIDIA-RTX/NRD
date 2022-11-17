/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

groupshared float4 s_Normal_MinHitDist[ BUFFER_Y ][ BUFFER_X ];

void Preload( uint2 sharedPos, int2 globalPos )
{
    globalPos = clamp( globalPos, 0, gRectSize - 1.0 );
    uint2 globalIdUser = gRectOrigin + globalPos;

    float4 temp = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ globalIdUser ] );

    #ifdef REBLUR_SPECULAR
        #ifdef REBLUR_OCCLUSION
            uint shift = gSpecCheckerboard != 2 ? 1 : 0;
            uint2 pos = uint2( globalPos.x >> shift, globalPos.y ) + gRectOrigin;
        #else
            uint2 pos = globalPos + ( gIsPrepassEnabled ? 0 : gRectOrigin );
        #endif

        REBLUR_TYPE spec = gIn_Spec[ pos ];
        float hitDistForTracking = ExtractHitDist( spec );
        #ifndef REBLUR_OCCLUSION
            hitDistForTracking = gSpecPrepassBlurRadius != 0.0 ? gIn_Spec_MinHitDist[ globalPos ] : hitDistForTracking;
        #endif
        temp.w = hitDistForTracking;
    #endif

    s_Normal_MinHitDist[ sharedPos.y ][ sharedPos.x ] = temp;
}

[numthreads( GROUP_X, GROUP_Y, 1 )]
NRD_EXPORT void NRD_CS_MAIN( int2 threadPos : SV_GroupThreadId, int2 pixelPos : SV_DispatchThreadId, uint threadIndex : SV_GroupIndex )
{
    uint2 pixelPosUser = gRectOrigin + pixelPos;
    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvRectSize;

    PRELOAD_INTO_SMEM;

    // Early out
    float viewZ = abs( gIn_ViewZ[ pixelPosUser ] );

    [branch]
    if( viewZ > gDenoisingRange )
        return;

    // Current position
    float3 Xv = STL::Geometry::ReconstructViewPosition( pixelUv, gFrustum, viewZ, gOrthoMode );
    float3 X = STL::Geometry::RotateVector( gViewToWorld, Xv );

    // Analyze neighbors
    int2 smemPos = threadPos + BORDER;
    float4 t = s_Normal_MinHitDist[ smemPos.y ][ smemPos.x ];
    float3 Navg = t.xyz;
    float hitDistForTracking = t.w;

    [unroll]
    for( j = 0; j <= BORDER * 2; j++ )
    {
        [unroll]
        for( i = 0; i <= BORDER * 2; i++ )
        {
            if( i == BORDER && j == BORDER )
                continue;

            int2 pos = threadPos + int2( i, j );
            float4 t = s_Normal_MinHitDist[ pos.y ][ pos.x ];

            Navg.xyz += t.xyz;

            #ifdef REBLUR_SPECULAR
                hitDistForTracking = min( hitDistForTracking, t.w ); // just "min" here works better than code from PrePass
            #endif
        }
    }

    Navg /= ( BORDER * 2 + 1 ) * ( BORDER * 2 + 1 ); // needs to be unnormalized!

    // Normal and roughness
    float materialID;
    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ pixelPosUser ], materialID );
    float3 N = normalAndRoughness.xyz;
    float roughness = normalAndRoughness.w;

    #ifdef REBLUR_SPECULAR
        float roughnessModified = STL::Filtering::GetModifiedRoughnessFromNormalVariance( roughness, Navg ); // TODO: needed?
    #endif

    // Previous position and surface motion uv
    float3 mv = gIn_Mv[ pixelPosUser ] * gMvScale;
    float3 Xprev = X;

    float2 smbPixelUv = pixelUv + mv.xy;
    if( gIsWorldSpaceMotionEnabled )
    {
        Xprev += mv;
        smbPixelUv = STL::Geometry::GetScreenUv( gWorldToClipPrev, Xprev );
    }
    else if( gMvScale.z != 0.0 )
    {
        float viewZprev = viewZ + mv.z;
        float3 Xvprevlocal = STL::Geometry::ReconstructViewPosition( smbPixelUv, gFrustumPrev, viewZprev, gOrthoMode ); // TODO: use gOrthoModePrev

        Xprev = STL::Geometry::RotateVectorInverse( gWorldToViewPrev, Xvprevlocal ) + gCameraDelta;
    }

    // Parallax
    float smbParallax = ComputeParallax( Xprev - gCameraDelta, gOrthoMode == 0.0 ? pixelUv : smbPixelUv, gWorldToClip, gRectSize, gUnproject, gOrthoMode );
    float smbParallaxInPixels = GetParallaxInPixels( smbParallax, gUnproject );

    // Curvature estimation along predicted motion ( tests 15, 40, 76, 133, 146, 147, 148 )
    float curvature = 0;
    #ifdef REBLUR_SPECULAR
    {
        float2 motionUv = STL::Geometry::GetScreenUv( gWorldToClip, Xprev - gCameraDelta, false );
        float2 cameraMotion2d = ( motionUv - pixelUv ) * gRectSize;
        cameraMotion2d /= max( length( cameraMotion2d ), 1.0 / ( 1.0 + gMaxAccumulatedFrameNum ) );
        cameraMotion2d *= gInvRectSize;

        // Low parallax - bilinear ( SMEM )
        float2 uv = pixelUv + cameraMotion2d * 0.5;
        STL::Filtering::Bilinear f = STL::Filtering::GetBilinearFilter( uv, gRectSize );

        int2 pos = threadPos + BORDER + int2( f.origin ) - pixelPos;
        pos = clamp( pos, 0, int2( BUFFER_X, BUFFER_Y ) - 2 );

        float3 n00 = s_Normal_MinHitDist[ pos.y ][ pos.x ].xyz;
        float3 n10 = s_Normal_MinHitDist[ pos.y ][ pos.x + 1 ].xyz;
        float3 n01 = s_Normal_MinHitDist[ pos.y + 1 ][ pos.x ].xyz;
        float3 n11 = s_Normal_MinHitDist[ pos.y + 1 ][ pos.x + 1 ].xyz;

        float3 n = STL::Filtering::ApplyBilinearFilter( n00, n10, n01, n11, f );
        n = normalize( n );

        // High parallax - nearest ( fetch )
        float2 uvHigh = gRectOffset + pixelUv + cameraMotion2d * smbParallaxInPixels;
        float3 nHigh = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness.SampleLevel( gNearestClamp, uvHigh * gResolutionScale, 0 ) ).xyz;
        float zHigh = abs( gIn_ViewZ.SampleLevel( gNearestClamp, uvHigh * gResolutionScale, 0 ) );

        float zError = abs( zHigh - viewZ ) * rcp( max( zHigh, viewZ ) );
        bool cmp = smbParallaxInPixels > 1.0 && zError < 0.1 && IsInScreen( uvHigh );

        uv = cmp ? uvHigh : uv;
        n = cmp ? nHigh : n;

        // Estimate curvature
        float3 xv = STL::Geometry::ReconstructViewPosition( uv, gFrustum, 1.0, gOrthoMode );
        float3 x = STL::Geometry::RotateVector( gViewToWorld, xv );
        float3 v = GetViewVector( x );

        // Values below this threshold get turned into garbage due to numerical imprecision
        float d = STL::Math::ManhattanDistance( N, n );
        float s = STL::Math::LinearStep( NRD_NORMAL_ENCODING_ERROR, 2.0 * NRD_NORMAL_ENCODING_ERROR, d );

        curvature = EstimateCurvature( n, v, N, X ) * s;
    }
    #endif

    // Previous viewZ ( 4x4, surface motion )
    /*
          Gather      => CatRom12    => Bilinear
        0x 0y 1x 1y       0y 1x
        0z 0w 1z 1w    0z 0w 1z 1w       0w 1z
        2x 2y 3x 3y    2x 2y 3x 3y       2y 3x
        2z 2w 3z 3w       2w 3z

         CatRom12     => Bilinear
           0x 1x
        0y 0z 1y 1z       0z 1y
        2x 2y 3x 3y       2y 3x
           2z 3z
    */
    STL::Filtering::CatmullRom smbCatromFilter = STL::Filtering::GetCatmullRomFilter( saturate( smbPixelUv ), gRectSizePrev );
    float2 smbCatromGatherUv = smbCatromFilter.origin * gInvScreenSize;
    float4 smbViewZ0 = gIn_Prev_ViewZ.GatherRed( gNearestClamp, smbCatromGatherUv, float2( 1, 1 ) ).wzxy;
    float4 smbViewZ1 = gIn_Prev_ViewZ.GatherRed( gNearestClamp, smbCatromGatherUv, float2( 3, 1 ) ).wzxy;
    float4 smbViewZ2 = gIn_Prev_ViewZ.GatherRed( gNearestClamp, smbCatromGatherUv, float2( 1, 3 ) ).wzxy;
    float4 smbViewZ3 = gIn_Prev_ViewZ.GatherRed( gNearestClamp, smbCatromGatherUv, float2( 3, 3 ) ).wzxy;

    float3 prevViewZ0 = UnpackViewZ( smbViewZ0.yzw );
    float3 prevViewZ1 = UnpackViewZ( smbViewZ1.xzw );
    float3 prevViewZ2 = UnpackViewZ( smbViewZ2.xyw );
    float3 prevViewZ3 = UnpackViewZ( smbViewZ3.xyz );

    // Previous normal averaged for all pixels in 2x2 footprint
    // IMPORTANT: bilinear filter can touch sky pixels, due to this reason "Post Blur" writes special values into sky-pixels
    STL::Filtering::Bilinear smbBilinearFilter = STL::Filtering::GetBilinearFilter( saturate( smbPixelUv ), gRectSizePrev );

    float2 smbBilinearGatherUv = ( smbBilinearFilter.origin + 1.0 ) * gInvScreenSize;
    float3 prevNavg = UnpackNormalAndRoughness( gIn_Prev_Normal_Roughness.SampleLevel( gLinearClamp, smbBilinearGatherUv, 0 ), false ).xyz;
    prevNavg = STL::Geometry::RotateVector( gWorldPrevToWorld, prevNavg );

    // Previous accum speed and materialID // TODO: 4x4 materialID footprint is reduced to 2x2 only
    uint4 smbInternalData = gIn_Prev_InternalData.GatherRed( gNearestClamp, smbBilinearGatherUv ).wzxy;

    float3 internalData00 = UnpackInternalData( smbInternalData.x );
    float3 internalData10 = UnpackInternalData( smbInternalData.y );
    float3 internalData01 = UnpackInternalData( smbInternalData.z );
    float3 internalData11 = UnpackInternalData( smbInternalData.w );

    float4 diffAccumSpeeds = float4( internalData00.x, internalData10.x, internalData01.x, internalData11.x );
    float4 specAccumSpeeds = float4( internalData00.y, internalData10.y, internalData01.y, internalData11.y );
    float4 prevMaterialIDs = float4( internalData00.z, internalData10.z, internalData01.z, internalData11.z );

    // Disocclsuion threshold
    float disocclusionThreshold = gDisocclusionThreshold;
    if( gHasDisocclusionThresholdMix )
        disocclusionThreshold = lerp( gDisocclusionThreshold, gDisocclusionThresholdAlternate, gIn_DisocclusionThresholdMix[ pixelPosUser ] );

    // Plane distance based disocclusion for surface motion
    float3 V = GetViewVector( X );
    float NoV = abs( dot( N, V ) );
    float pixelSize = PixelRadiusToWorld( gUnproject, gOrthoMode, 1.0, viewZ );
    float frustumHeight = pixelSize * gRectSize.y;

    float smbDisocclusionThreshold = gDisocclusionThreshold * frustumHeight / lerp( NoV, 1.0, saturate( smbParallaxInPixels / 30.0 ) );

    float mvLengthFactor = STL::Math::LinearStep( 0.5, 1.0, smbParallaxInPixels );
    float frontFacing = lerp( cos( STL::Math::DegToRad( 135.0 ) ), cos( STL::Math::DegToRad( 91.0 ) ), mvLengthFactor );
    bool isInScreenAndNotBackfacing = IsInScreen( smbPixelUv ) && dot( prevNavg, Navg ) > frontFacing;
    smbDisocclusionThreshold = isInScreenAndNotBackfacing ? smbDisocclusionThreshold : -1.0;

    float3 Xvprev = STL::Geometry::AffineTransform( gWorldToViewPrev, Xprev );
    float3 smbPlaneDist0 = abs( prevViewZ0 - Xvprev.z );
    float3 smbPlaneDist1 = abs( prevViewZ1 - Xvprev.z );
    float3 smbPlaneDist2 = abs( prevViewZ2 - Xvprev.z );
    float3 smbPlaneDist3 = abs( prevViewZ3 - Xvprev.z );
    float3 smbOcclusion0 = step( smbPlaneDist0, smbDisocclusionThreshold );
    float3 smbOcclusion1 = step( smbPlaneDist1, smbDisocclusionThreshold );
    float3 smbOcclusion2 = step( smbPlaneDist2, smbDisocclusionThreshold );
    float3 smbOcclusion3 = step( smbPlaneDist3, smbDisocclusionThreshold );

    float4 smbOcclusionWeights = STL::Filtering::GetBilinearCustomWeights( smbBilinearFilter, float4( smbOcclusion0.z, smbOcclusion1.y, smbOcclusion2.y, smbOcclusion3.x ) );
    bool smbAllowCatRom = dot( smbOcclusion0 + smbOcclusion1 + smbOcclusion2 + smbOcclusion3, 1.0 ) > 11.5 && REBLUR_USE_CATROM_FOR_SURFACE_MOTION_IN_TA;
    #ifdef REBLUR_DIRECTIONAL_OCCLUSION
        smbAllowCatRom = false;
    #endif

    float smbFootprintQuality = STL::Filtering::ApplyBilinearFilter( smbOcclusion0.z, smbOcclusion1.y, smbOcclusion2.y, smbOcclusion3.x, smbBilinearFilter );
    smbFootprintQuality = STL::Math::Sqrt01( smbFootprintQuality );

    // Material ID check
    float4 materialCmps = CompareMaterials( materialID, prevMaterialIDs, 1 );
    smbOcclusion0.z *= materialCmps.x;
    smbOcclusion1.y *= materialCmps.y;
    smbOcclusion2.y *= materialCmps.z;
    smbOcclusion3.x *= materialCmps.w;

    float4 smbOcclusionWeightsWithMaterialID = STL::Filtering::GetBilinearCustomWeights( smbBilinearFilter, float4( smbOcclusion0.z, smbOcclusion1.y, smbOcclusion2.y, smbOcclusion3.x ) );
    bool smbAllowCatRomWithMaterialID = smbAllowCatRom && dot( materialCmps, 1.0 ) > 3.5 && REBLUR_USE_CATROM_FOR_SURFACE_MOTION_IN_TA;

    float smbFootprintQualityWithMaterialID = STL::Filtering::ApplyBilinearFilter( smbOcclusion0.z, smbOcclusion1.y, smbOcclusion2.y, smbOcclusion3.x, smbBilinearFilter );
    smbFootprintQualityWithMaterialID = STL::Math::Sqrt01( smbFootprintQualityWithMaterialID );

    // Avoid footprint momentary stretching due to changed viewing angle
    float3 smbVprev = GetViewVectorPrev( Xprev, gCameraDelta );
    float NoVprev = abs( dot( N, smbVprev ) ); // TODO: should be prevNavg ( normalized? ), but jittering breaks logic
    float sizeQuality = ( NoVprev + 1e-3 ) / ( NoV + 1e-3 ); // this order because we need to fix stretching only, shrinking is OK
    sizeQuality *= sizeQuality;
    sizeQuality = lerp( 0.1, 1.0, saturate( sizeQuality ) );
    smbFootprintQuality *= sizeQuality;
    smbFootprintQualityWithMaterialID *= sizeQuality;

    // Bits
    float fbits = smbAllowCatRom * 2.0;
    fbits += smbOcclusion0.z * 4.0 + smbOcclusion1.y * 8.0 + smbOcclusion2.y * 16.0 + smbOcclusion3.x * 32.0;

    // Update accumulation speeds
    #ifdef REBLUR_DIFFUSE
        float4 diffOcclusionWeights = gDiffMaterialMask ? smbOcclusionWeightsWithMaterialID : smbOcclusionWeights;
        float diffHistoryConfidence = gDiffMaterialMask ? smbFootprintQualityWithMaterialID : smbFootprintQuality;
        bool diffAllowCatRom = gDiffMaterialMask ? smbAllowCatRomWithMaterialID : smbAllowCatRom;

        if( gHasHistoryConfidence )
            diffHistoryConfidence *= gIn_Diff_Confidence[ pixelPosUser ];

        float diffAccumSpeed = STL::Filtering::ApplyBilinearCustomWeights( diffAccumSpeeds.x, diffAccumSpeeds.y, diffAccumSpeeds.z, diffAccumSpeeds.w, diffOcclusionWeights );
        diffAccumSpeed *= lerp( diffHistoryConfidence, 1.0, 1.0 / ( 1.0 + diffAccumSpeed ) );
    #endif

    #ifdef REBLUR_SPECULAR
        float4 specOcclusionWeights = gSpecMaterialMask ? smbOcclusionWeightsWithMaterialID : smbOcclusionWeights;
        float specHistoryConfidence = gSpecMaterialMask ? smbFootprintQualityWithMaterialID : smbFootprintQuality;
        bool specAllowCatRom = gSpecMaterialMask ? smbAllowCatRomWithMaterialID : smbAllowCatRom;

        if( gHasHistoryConfidence )
            specHistoryConfidence *= gIn_Spec_Confidence[ pixelPosUser ];

        float specAccumSpeed = STL::Filtering::ApplyBilinearCustomWeights( specAccumSpeeds.x, specAccumSpeeds.y, specAccumSpeeds.z, specAccumSpeeds.w, specOcclusionWeights );
        specAccumSpeed *= lerp( specHistoryConfidence, 1.0, 1.0 / ( 1.0 + specAccumSpeed ) );
    #endif

    uint checkerboard = STL::Sequence::CheckerBoard( pixelPos, gFrameIndex );
    #ifdef REBLUR_OCCLUSION
        int3 checkerboardPos = pixelPosUser.xyx + int3( -1, 0, 1 );
        float viewZ0 = abs( gIn_ViewZ[ checkerboardPos.xy ] );
        float viewZ1 = abs( gIn_ViewZ[ checkerboardPos.zy ] );
        float2 wc = GetBilateralWeight( float2( viewZ0, viewZ1 ), viewZ );
        wc *= STL::Math::PositiveRcp( wc.x + wc.y );
    #endif

    // Diffuse
    #ifdef REBLUR_DIFFUSE
        bool diffHasData = gDiffCheckerboard == 2 || checkerboard == gDiffCheckerboard;
        #ifdef REBLUR_OCCLUSION
            uint diffShift = gDiffCheckerboard != 2 ? 1 : 0;
            uint2 diffPos = uint2( pixelPos.x >> diffShift, pixelPos.y ) + gRectOrigin;
        #else
            uint2 diffPos = pixelPos;
        #endif

        REBLUR_TYPE diff = gIn_Diff[ diffPos ];
        #ifdef REBLUR_SH
            float4 diffSh = gIn_DiffSh[ diffPos ];
        #endif

        // Sample history - surface motion
        REBLUR_TYPE smbDiffHistory;
        float4 smbDiffShHistory;
        float smbDiffFastHistory;
        BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights(
            saturate( smbPixelUv ) * gRectSizePrev, gInvScreenSize,
            diffOcclusionWeights, diffAllowCatRom,
            gIn_Diff_History, smbDiffHistory
            #ifdef REBLUR_SH
                , gIn_DiffSh_History, smbDiffShHistory
            #endif
            #if( REBLUR_USE_FAST_HISTORY == 1 && !defined( REBLUR_OCCLUSION ) )
                , gIn_DiffFast_History, smbDiffFastHistory
            #endif
        );

        // Avoid negative values
        smbDiffHistory = ClampNegativeToZero( smbDiffHistory );

        // Accumulation with checkerboard resolve // TODO: materialID support?
        #ifdef REBLUR_OCCLUSION
            float d0 = gIn_Diff[ uint2( ( pixelPos.x - 1 ) >> diffShift, pixelPos.y ) + gRectOrigin ];
            float d1 = gIn_Diff[ uint2( ( pixelPos.x + 1 ) >> diffShift, pixelPos.y ) + gRectOrigin ];

            if( !diffHasData )
            {
                diff *= saturate( 1.0 - wc.x - wc.y );
                diff += d0 * wc.x + d1 * wc.y;
            }
        #endif

        float diffNonLinearAccumSpeed = 1.0 / ( 1.0 + diffAccumSpeed );
        if( !diffHasData )
            diffNonLinearAccumSpeed *= lerp( 1.0 - gCheckerboardResolveAccumSpeed, 1.0, diffNonLinearAccumSpeed );

        REBLUR_TYPE diffResult = MixHistoryAndCurrent( smbDiffHistory, diff, diffNonLinearAccumSpeed );
        #ifdef REBLUR_SH
            float4 diffShResult = MixHistoryAndCurrent( smbDiffShHistory, diffSh, diffNonLinearAccumSpeed );
        #endif

        // Anti-firefly suppressor
        float diffAntifireflyFactor = diffAccumSpeed * gBlurRadius * REBLUR_FIREFLY_SUPPRESSOR_RADIUS_SCALE;
        diffAntifireflyFactor /= 1.0 + diffAntifireflyFactor;

        float diffHitDistResult = ExtractHitDist( diffResult );
        float diffHitDistClamped = min( diffHitDistResult, ExtractHitDist( smbDiffHistory ) * REBLUR_MAX_FIREFLY_RELATIVE_INTENSITY.y );
        diffHitDistClamped = lerp( diffHitDistResult, diffHitDistClamped, diffAntifireflyFactor );

        #if( defined REBLUR_OCCLUSION || defined REBLUR_DIRECTIONAL_OCCLUSION )
            diffResult = ChangeLuma( diffResult, diffHitDistClamped );
        #else
            float diffLumaResult = GetLuma( diffResult );
            float diffLumaClamped = min( diffLumaResult, GetLuma( smbDiffHistory ) * REBLUR_MAX_FIREFLY_RELATIVE_INTENSITY.x );
            diffLumaClamped = lerp( diffLumaResult, diffLumaClamped, diffAntifireflyFactor );

            diffResult = ChangeLuma( diffResult, diffLumaClamped );
            diffResult.w = diffHitDistClamped;

            #ifdef REBLUR_SH
                diffShResult.xyz *= GetLumaScale( length( diffShResult.xyz ), diffLumaClamped );
            #endif
        #endif

        // Output
        float diffError = GetColorErrorForAdaptiveRadiusScale( diffResult, smbDiffHistory, diffAccumSpeed );

        gOut_Diff[ pixelPos ] = diffResult;
        #ifdef REBLUR_SH
            gOut_DiffSh[ pixelPos ] = diffShResult;
        #endif

        // Fast history
        #if( REBLUR_USE_FAST_HISTORY == 1 && !defined( REBLUR_OCCLUSION ) )
            smbDiffFastHistory = diffAccumSpeed < gMaxFastAccumulatedFrameNum ? GetLuma( smbDiffHistory ) : smbDiffFastHistory;

            float diffFastAccumSpeed = min( diffAccumSpeed, gMaxFastAccumulatedFrameNum );
            float diffFastNonLinearAccumSpeed = 1.0 / ( 1.0 + diffFastAccumSpeed );
            if( !diffHasData )
                diffFastNonLinearAccumSpeed *= lerp( 1.0 - gCheckerboardResolveAccumSpeed, 1.0, diffFastNonLinearAccumSpeed );

            float diffFastResult = MixFastHistoryAndCurrent( smbDiffFastHistory, GetLuma( diff ), diffFastNonLinearAccumSpeed );

            gOut_DiffFast[ pixelPos ] = diffFastResult;
        #endif
    #else
        float diffAccumSpeed = 0;
        float diffError = 0;
    #endif

    // Specular
    #ifdef REBLUR_SPECULAR
        bool specHasData = gSpecCheckerboard == 2 || checkerboard == gSpecCheckerboard;
        #ifdef REBLUR_OCCLUSION
            uint specShift = gSpecCheckerboard != 2 ? 1 : 0;
            uint2 specPos = uint2( pixelPos.x >> specShift, pixelPos.y ) + gRectOrigin;
        #else
            uint2 specPos = pixelPos;
        #endif

        REBLUR_TYPE spec = gIn_Spec[ specPos ];
        #ifdef REBLUR_SH
            float4 specSh = gIn_SpecSh[ specPos ];
        #endif

        // Checkerboard resolve // TODO: materialID support?
        #ifdef REBLUR_OCCLUSION
            float s0 = gIn_Spec[ uint2( ( pixelPos.x - 1 ) >> specShift, pixelPos.y ) + gRectOrigin ];
            float s1 = gIn_Spec[ uint2( ( pixelPos.x + 1 ) >> specShift, pixelPos.y ) + gRectOrigin ];

            if( !specHasData )
            {
                spec *= saturate( 1.0 - wc.x - wc.y );
                spec += s0 * wc.x + s1 * wc.y;
            }
        #endif

        // Hit distance for tracking ( tests 8, 110, 139 )
        float hitDistScale = _REBLUR_GetHitDistanceNormalization( viewZ, gHitDistParams, roughness );
        hitDistForTracking *= hitDistScale; // TODO: clamp to +/- 3 sigma?

        // Virtual motion
        float4 D = STL::ImportanceSampling::GetSpecularDominantDirection( N, V, roughness, STL_SPECULAR_DOMINANT_DIRECTION_G2 );

        float3 Xvirtual = GetXvirtual( NoV, hitDistForTracking, curvature, X, Xprev, V, D.w );
        float2 vmbPixelUv = STL::Geometry::GetScreenUv( gWorldToClipPrev, Xvirtual, false );

        float2 vmbDelta = vmbPixelUv - smbPixelUv;
        float vmbPixelsTraveled = length( vmbDelta * gRectSize );

        // Adjust curvature if curvature sign oscillation is forseen // TODO: is there a better way? fix curvature?
        float curvatureCorrectionThreshold = smbParallaxInPixels + gInvRectSize.x;
        float curvatureCorrection = STL::Math::SmoothStep( 1.05 * curvatureCorrectionThreshold, 0.95 * curvatureCorrectionThreshold, vmbPixelsTraveled );
        curvature *= curvatureCorrection;

        Xvirtual = GetXvirtual( NoV, hitDistForTracking, curvature, X, Xprev, V, D.w );
        vmbPixelUv = STL::Geometry::GetScreenUv( gWorldToClipPrev, Xvirtual, false );

        float XvirtualLength = length( Xvirtual );

        // Update after curvature correction
        vmbDelta = vmbPixelUv - smbPixelUv;
        vmbPixelsTraveled = length( vmbDelta * gRectSize );

        // Virtual history amount - base
        float virtualHistoryAmount = IsInScreen( vmbPixelUv );
        virtualHistoryAmount *= 1.0 - gReference;
        virtualHistoryAmount *= D.w;

        // Virtual motion amount - surface
        STL::Filtering::Bilinear vmbBilinearFilter = STL::Filtering::GetBilinearFilter( saturate( vmbPixelUv ), gRectSizePrev );
        float2 vmbBilinearGatherUv = ( vmbBilinearFilter.origin + 1.0 ) * gInvScreenSize;
        float4 vmbViewZs = UnpackViewZ( gIn_Prev_ViewZ.GatherRed( gNearestClamp, vmbBilinearGatherUv ).wzxy );
        float3 vmbVv = STL::Geometry::ReconstructViewPosition( vmbPixelUv, gFrustumPrev, 1.0 ); // unnormalized, orthoMode = 0
        float3 Nvprev = STL::Geometry::RotateVector( gWorldToViewPrev, N );
        float NoXreal = dot( N, X - gCameraDelta );
        float4 NoX = ( Nvprev.x * vmbVv.x + Nvprev.y * vmbVv.y ) * ( gOrthoMode == 0 ? vmbViewZs : gOrthoMode ) + Nvprev.z * vmbVv.z * vmbViewZs;
        float4 vmbPlaneDist = abs( NoX - NoXreal ) / frustumHeight;
        float4 vmbOcclusion = step( vmbPlaneDist, gDisocclusionThreshold );

        float vmbFootprintQuality = STL::Filtering::ApplyBilinearFilter( vmbOcclusion.x, vmbOcclusion.y, vmbOcclusion.z, vmbOcclusion.w, vmbBilinearFilter );
        vmbFootprintQuality = STL::Math::Sqrt01( vmbFootprintQuality );
        virtualHistoryAmount *= vmbFootprintQuality;

        bool vmbAllowCatRom = dot( vmbOcclusion, 1.0 ) > 3.5 && REBLUR_USE_CATROM_FOR_VIRTUAL_MOTION_IN_TA;
        vmbAllowCatRom = vmbAllowCatRom && specAllowCatRom; // helps to reduce over-sharpening in disoccluded areas

        // Virtual motion based accumulation speed
        uint4 vmbInternalData = gIn_Prev_InternalData.GatherRed( gNearestClamp, vmbBilinearGatherUv ).wzxy;

        float3 vmbInternalData00 = UnpackInternalData( vmbInternalData.x );
        float3 vmbInternalData10 = UnpackInternalData( vmbInternalData.y );
        float3 vmbInternalData01 = UnpackInternalData( vmbInternalData.z );
        float3 vmbInternalData11 = UnpackInternalData( vmbInternalData.w );

        float4 vmbOcclusionWeights = STL::Filtering::GetBilinearCustomWeights( vmbBilinearFilter, vmbOcclusion );
        float vmbSpecAccumSpeed = STL::Filtering::ApplyBilinearCustomWeights( vmbInternalData00.y, vmbInternalData10.y, vmbInternalData01.y, vmbInternalData11.y, vmbOcclusionWeights );
        vmbSpecAccumSpeed *= lerp( vmbFootprintQuality, 1.0, 1.0 / ( 1.0 + vmbSpecAccumSpeed ) );

        float responsiveAccumulationAmount = GetResponsiveAccumulationAmount( roughness );
        responsiveAccumulationAmount = lerp( 1.0, GetSpecMagicCurve( roughness ), responsiveAccumulationAmount );

        float vmbMaxFrameNum = GetFPS( ) * responsiveAccumulationAmount;
        vmbSpecAccumSpeed = min( vmbSpecAccumSpeed, vmbMaxFrameNum );

        // Virtual motion amount - normal
        float lobeHalfAngle = STL::ImportanceSampling::GetSpecularLobeHalfAngle( roughness );
        lobeHalfAngle = max( lobeHalfAngle, REBLUR_NORMAL_ULP );

        // Estimate how many pixels are traveled by virtual motion - how many radians can it be?
        // If curvature angle is multiplied by path length then we can get an angle exceeding 2 * PI, what is impossible. The max
        // angle is PI ( most left and most right points on a hemisphere ), it can be achieved by using "tan" instead of angle.
        float curvatureAngleTan = pixelSize * abs( curvature ); // tana = pixelSize / curvatureRadius = pixelSize * curvature
        curvatureAngleTan *= 1.0 + vmbPixelsTraveled / max( NoV, 0.01 ); // path length
        curvatureAngleTan *= gFramerateScale; // "per frame" to "per sec"

        float curvatureAngle = atan( curvatureAngleTan );

        float4 vmbNormalAndRoughness = UnpackNormalAndRoughness( gIn_Prev_Normal_Roughness.SampleLevel( gLinearClamp, vmbPixelUv * gRectSizePrev * gInvScreenSize, 0 ) );
        float3 vmbN = STL::Geometry::RotateVector( gWorldPrevToWorld, vmbNormalAndRoughness.xyz );

        float angle = lobeHalfAngle + curvatureAngle;
        float virtualHistoryNormalBasedConfidence = GetEncodingAwareNormalWeight( N, vmbN, angle );
        virtualHistoryAmount *= lerp( 1.0 - saturate( vmbPixelsTraveled ), 1.0, virtualHistoryNormalBasedConfidence ); // jitter friendly

        // Virtual motion amount - roughness
        float roughnessFraction = lerp( 0.2, 1.0, STL::BRDF::Pow5( NoV ) );
        float virtualHistoryRoughnessBasedConfidence = GetEncodingAwareRoughnessWeights( roughness, vmbNormalAndRoughness.w, roughnessFraction );
        virtualHistoryAmount *= lerp( 1.0 - saturate( vmbPixelsTraveled ), 1.0, virtualHistoryRoughnessBasedConfidence ); // jitter friendly

        // Sample history - surface motion
        REBLUR_TYPE smbSpecHistory;
        float4 smbSpecShHistory;
        float smbSpecFastHistory;
        BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights(
            saturate( smbPixelUv ) * gRectSizePrev, gInvScreenSize,
            specOcclusionWeights, specAllowCatRom,
            gIn_Spec_History, smbSpecHistory
            #ifdef REBLUR_SH
                , gIn_SpecSh_History, smbSpecShHistory
            #endif
            #if( REBLUR_USE_FAST_HISTORY == 1 && !defined( REBLUR_OCCLUSION ) )
                , gIn_SpecFast_History, smbSpecFastHistory
            #endif
        );

        // Sample history - virtual motion
        REBLUR_TYPE vmbSpecHistory;
        float4 vmbSpecShHistory;
        float vmbSpecFastHistory;
        BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights(
            saturate( vmbPixelUv ) * gRectSizePrev, gInvScreenSize,
            vmbOcclusionWeights, vmbAllowCatRom,
            gIn_Spec_History, vmbSpecHistory
            #ifdef REBLUR_SH
                , gIn_SpecSh_History, vmbSpecShHistory
            #endif
            #if( REBLUR_USE_FAST_HISTORY == 1 && !defined( REBLUR_OCCLUSION ) )
                , gIn_SpecFast_History, vmbSpecFastHistory
            #endif
        );

        // Avoid negative values
        smbSpecHistory = ClampNegativeToZero( smbSpecHistory );
        vmbSpecHistory = ClampNegativeToZero( vmbSpecHistory );

        // Virtual motion confidence - virtual parallax difference
        // Tests 3, 6, 8, 11, 14, 100, 103, 104, 106, 109, 110, 114, 120, 127, 130, 131, 132, 138, 139 and 9e
        float smc = GetSpecMagicCurve( roughnessModified );
        float smbSpecAccumSpeedFactor = lerp( 5.0, 1.0, smc );
        smbSpecAccumSpeedFactor *= lerp( 1.0, lerp( 1.0, 0.25, smc ), NoV ); // TODO: potentially needs to be tuned
        smbSpecAccumSpeedFactor *= gFramerateScale;

        float smbSpecAccumSpeed = GetSmbAccumSpeed( smbSpecAccumSpeedFactor, vmbPixelsTraveled, viewZ, specAccumSpeed, angle );

        // Current hit distance estimation: // TODO: try to use hit distances processed by OCCLUSION denoiser
        // - can't use input, because it's noisy ( but it's not noisy for 0 roughness and we must use input in this case )
        // - for glossy reflections prefer smb-based hit distance, because it's already denoised
        // - combine with a small amount of vmb-based hit distance to further reduce noise
        float smbHitDist = lerp( ExtractHitDist( spec ), ExtractHitDist( smbSpecHistory ), STL::Math::SmoothStep( 0.04, 0.11, roughnessModified ) * smbSpecAccumSpeed / ( 1.0 + smbSpecAccumSpeed ) );
        smbHitDist = lerp( smbHitDist, ExtractHitDist( vmbSpecHistory ), 0.5 * vmbSpecAccumSpeed / ( 1.0 + vmbSpecAccumSpeed ) );
        smbHitDist *= hitDistScale;

        float vmbHitDist = ExtractHitDist( vmbSpecHistory );
        vmbHitDist *= hitDistScale; // we could use "vmbViewZs" and "vmbNormalAndRoughness.w" but there are dedicated weights for it

        float3 smbXvirtual = GetXvirtual( NoV, smbHitDist, curvature, X, Xprev, V, D.w );
        float2 smbUv = STL::Geometry::GetScreenUv( gWorldToClipPrev, smbXvirtual, false );
        float3 vmbXvirtual = GetXvirtual( NoV, vmbHitDist, curvature, X, Xprev, V, D.w );
        float2 vmbUv = STL::Geometry::GetScreenUv( gWorldToClipPrev, vmbXvirtual, false );

        float minLobeTanHalfAngle = 0.5 * gInvRectSize.x;
        float lobeTanHalfAngle = STL::ImportanceSampling::GetSpecularLobeTanHalfAngle( roughness, 0.2 ); // why 20%?
        float lobeRadius = hitDistForTracking * max( lobeTanHalfAngle, minLobeTanHalfAngle ); // no NoD to avoid losing history under very glancing angles
        float lobeRadiusInPixels = lobeRadius / PixelRadiusToWorld( gUnproject, gOrthoMode, 1.0, XvirtualLength );
        lobeRadiusInPixels += 0.5 * smc;

        float deltaParallaxInPixels = length( ( smbUv - vmbUv ) * gRectSize );
        float virtualHistoryParallaxBasedConfidence = STL::Math::SmoothStep( lobeRadiusInPixels, 0.0, deltaParallaxInPixels );

        // Virtual motion confidence - fixing trails if radiance on a flat surface is taken from a sloppy surface
        vmbDelta *= STL::Math::Rsqrt( STL::Math::LengthSquared( vmbDelta ) );
        vmbDelta /= gRectSizePrev;
        vmbDelta *= saturate( vmbPixelsTraveled / 0.1 ) + vmbPixelsTraveled / REBLUR_VIRTUAL_MOTION_PREV_PREV_WEIGHT_ITERATION_NUM;

        float virtualHistoryPrevPrevBasedConfidence = 1.0;
        [unroll]
        for( i = 1; i <= REBLUR_VIRTUAL_MOTION_PREV_PREV_WEIGHT_ITERATION_NUM; i++ )
        {
            float2 vmbPixelUvPrev = vmbPixelUv + vmbDelta * i;
            float4 vmbNormalAndRoughnessPrev = UnpackNormalAndRoughness( gIn_Prev_Normal_Roughness.SampleLevel( gLinearClamp, vmbPixelUvPrev * gRectSizePrev * gInvScreenSize, 0 ) );

            float w = GetEncodingAwareNormalWeight( vmbNormalAndRoughness.xyz, vmbNormalAndRoughnessPrev.xyz, angle + curvatureAngle * i, curvatureAngle );
            float wr = GetEncodingAwareRoughnessWeights( vmbNormalAndRoughness.w, vmbNormalAndRoughnessPrev.w, roughnessFraction );
            w *= lerp( 0.33 * i, 1.0, lerp( 1.0 - saturate( abs( vmbPixelsTraveled ) ), 1.0, wr ) );

            float isOutOfScreen = 1.0 - IsInScreen( vmbPixelUvPrev );
            virtualHistoryPrevPrevBasedConfidence *= saturate( w + isOutOfScreen );
            virtualHistoryRoughnessBasedConfidence *= saturate( wr + isOutOfScreen );
        }

        // Virtual motion - accumulation acceleration
        float virtualHistoryConfidence = virtualHistoryPrevPrevBasedConfidence * virtualHistoryParallaxBasedConfidence;
        vmbSpecAccumSpeed *= virtualHistoryConfidence;

        // Surface motion ( test 9 )
        smbSpecAccumSpeedFactor *= lerp( 0.5 + 0.5 * virtualHistoryRoughnessBasedConfidence, 1.0, virtualHistoryAmount );

        smbSpecAccumSpeed = GetSmbAccumSpeed( smbSpecAccumSpeedFactor, vmbPixelsTraveled, viewZ, specAccumSpeed, angle );

        // Fallback to surface motion if virtual motion doesn't go well ( tests 103, 111, 132, e9, e11 )
        virtualHistoryAmount *= saturate( ( vmbSpecAccumSpeed + NRD_EPS ) / ( smbSpecAccumSpeed + NRD_EPS ) );

        // Accumulation with checkerboard resolve // TODO: materialID support?
        specAccumSpeed = lerp( smbSpecAccumSpeed, vmbSpecAccumSpeed, virtualHistoryAmount );

        float specNonLinearAccumSpeed = 1.0 / ( 1.0 + specAccumSpeed );
        if( !specHasData )
        {
            float confidence = 1.0 / ( 1.0 + specAccumSpeed );
            specNonLinearAccumSpeed *= lerp( 1.0 - gCheckerboardResolveAccumSpeed, 1.0, confidence );
        }

        REBLUR_TYPE specHistory = lerp( smbSpecHistory, vmbSpecHistory, virtualHistoryAmount );
        REBLUR_TYPE specResult = MixHistoryAndCurrent( specHistory, spec, specNonLinearAccumSpeed, roughnessModified );
        #ifdef REBLUR_SH
            float4 specShHistory = lerp( smbSpecShHistory, vmbSpecShHistory, virtualHistoryAmount );
            float4 specShResult = MixHistoryAndCurrent( specShHistory, specSh, specNonLinearAccumSpeed, roughnessModified );
        #endif

        // Anti-firefly suppressor
        float specAntifireflyFactor = specAccumSpeed * gBlurRadius * REBLUR_FIREFLY_SUPPRESSOR_RADIUS_SCALE * smc;
        specAntifireflyFactor /= 1.0 + specAntifireflyFactor;

        float specHitDistResult = ExtractHitDist( specResult );
        float specHitDistClamped = min( specHitDistResult, ExtractHitDist( specHistory ) * REBLUR_MAX_FIREFLY_RELATIVE_INTENSITY.y );
        specHitDistClamped = lerp( specHitDistResult, specHitDistClamped, specAntifireflyFactor );

        #if( defined REBLUR_OCCLUSION || defined REBLUR_DIRECTIONAL_OCCLUSION )
            specResult = ChangeLuma( specResult, specHitDistClamped );
        #else
            float specLumaResult = GetLuma( specResult );
            float specLumaClamped = min( specLumaResult, GetLuma( specHistory ) * REBLUR_MAX_FIREFLY_RELATIVE_INTENSITY.x );
            specLumaClamped = lerp( specLumaResult, specLumaClamped, specAntifireflyFactor );

            specResult = ChangeLuma( specResult, specLumaClamped );
            specResult.w = specHitDistClamped;

            #ifdef REBLUR_SH
                specShResult.xyz *= GetLumaScale( length( specShResult.xyz ), specLumaClamped );
            #endif
        #endif

        // Output
        float specError = GetColorErrorForAdaptiveRadiusScale( specResult, specHistory, specAccumSpeed, roughness );

        gOut_Spec[ pixelPos ] = specResult;
        #ifdef REBLUR_SH
            gOut_SpecSh[ pixelPos ] = specShResult;
        #endif

        // Fast history
        #if( REBLUR_USE_FAST_HISTORY == 1 && !defined( REBLUR_OCCLUSION ) )
            float specFastHistory = lerp( smbSpecFastHistory, vmbSpecFastHistory, virtualHistoryAmount );
            specFastHistory = specAccumSpeed < gMaxFastAccumulatedFrameNum ? GetLuma( specHistory ) : specFastHistory;

            float specFastAccumSpeed = min( specAccumSpeed, gMaxFastAccumulatedFrameNum );
            specFastAccumSpeed *= lerp( 1.0, virtualHistoryConfidence, virtualHistoryAmount );

            float specFastNonLinearAccumSpeed = 1.0 / ( 1.0 + specFastAccumSpeed );
            if( !specHasData )
                specFastNonLinearAccumSpeed *= lerp( 1.0 - gCheckerboardResolveAccumSpeed, 1.0, specFastNonLinearAccumSpeed );

            float specFastResult = MixFastHistoryAndCurrent( specFastHistory, GetLuma( spec ), specFastNonLinearAccumSpeed );

            gOut_SpecFast[ pixelPos ] = specFastResult;
        #endif
    #else
        float specAccumSpeed = 0;
        float virtualHistoryAmount = 0;
        float specError = 0;
    #endif

    // Output
    gOut_Data1[ pixelPos ] = PackData1( diffAccumSpeed, diffError, specAccumSpeed, specError );

    #ifndef REBLUR_OCCLUSION
        gOut_Data2[ pixelPos ] = PackData2( fbits, curvature, virtualHistoryAmount, viewZ );
    #endif
}
