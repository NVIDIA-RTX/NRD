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

#include "RELAX_Config.hlsli"
#include "RELAX_Validation.resources.hlsli"

#include "Common.hlsli"

#include "RELAX_Common.hlsli"

#define VIEWPORT_SIZE   0.25
#define OFFSET          5

/*
Viewport layout:
 0   1   2   3
 4   5   6   7
 8   9  10  11
12  13  14  15
*/

[numthreads( 16, 16, 1 )]
NRD_EXPORT void NRD_CS_MAIN( NRD_CS_MAIN_ARGS )
{
    NRD_CTA_ORDER_DEFAULT;

    if( any( pixelPos >= gRectSize ) )
        return;

    if( gResetHistory != 0 )
    {
        gOut_Validation[ pixelPos ] = 0;
        return;
    }

    float2 pixelUv = float2( pixelPos + 0.5 ) / gRectSize;

    float2 viewportUv = frac( pixelUv / VIEWPORT_SIZE );
    float2 viewportId = floor( pixelUv / VIEWPORT_SIZE );
    float viewportIndex = viewportId.y / VIEWPORT_SIZE + viewportId.x;

    float2 viewportUvScaled = viewportUv * gResolutionScale;
    int2 viewportPixelPos = int2( viewportUvScaled * gResourceSize );

    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ WithRectOrigin( viewportPixelPos ) ] );
    float viewZ = UnpackViewZ( gIn_ViewZ[ WithRectOrigin( viewportPixelPos ) ] );
    float3 mv = gIn_Mv[ WithRectOrigin( viewportPixelPos ) ] * gMvScale.xyz;

    float historyLength = 255.0 * gIn_HistoryLength[ viewportPixelPos ] - 1.0;

    float3 N = normalAndRoughness.xyz;
    float roughness = normalAndRoughness.w;

    float3 X = GetCurrentWorldPosFromClipSpaceXY( viewportUv * 2.0 - 1.0, abs( viewZ ) );

    bool isInf = !IsInDenoisingRange(abs( viewZ ));
    bool checkerboard = Sequence::CheckerBoard( pixelPos >> 2, 0 );

    uint4 textState = Text::Init( pixelPos, viewportId * gRectSize * VIEWPORT_SIZE + OFFSET, 1 );

    float4 result = gOut_Validation[ pixelPos ];

    if( viewportIndex == 4 )
    {
        // World-space normal
        Text::Print_ch( 'N', textState );
        Text::Print_ch( 'O', textState );
        Text::Print_ch( 'R', textState );
        Text::Print_ch( 'M', textState );
        Text::Print_ch( 'A', textState );
        Text::Print_ch( 'L', textState );
        Text::Print_ch( 'S', textState );
        Text::NextChar( textState );
        Text::Print_ui( NRD_NORMAL_ENCODING, textState );

        result.xyz = N * 0.5 + 0.5;
        result.w = 1.0;
    }
    else if( viewportIndex == 1 )
    {
        // Linear roughness
        Text::Print_ch( 'R', textState );
        Text::Print_ch( 'O', textState );
        Text::Print_ch( 'U', textState );
        Text::Print_ch( 'G', textState );
        Text::Print_ch( 'H', textState );
        Text::Print_ch( 'N', textState );
        Text::Print_ch( 'E', textState );
        Text::Print_ch( 'S', textState );
        Text::Print_ch( 'S', textState );
        Text::NextChar( textState );
        Text::Print_ui( NRD_ROUGHNESS_ENCODING, textState );

        result.xyz = normalAndRoughness.w;
        result.w = 1.0;
    }
    else if( viewportIndex == 2 )
    {
        // View Z
        Text::Print_ch( 'Z', textState );
        if( viewZ < 0 )
            Text::Print_ch( Text::Char_Minus, textState );

        float f = 0.1 * abs( viewZ ) / ( 1.0 + 0.1 * abs( viewZ ) ); // TODO: tuned for meters
        float3 color = viewZ < 0.0 ? float3( 0, 0, 1 ) : float3( 0, 1, 0 );

        result.xyz = isInf ? float3( 1, 0, 0 ) : color * f;
        result.w = 1.0;
    }
    else if( viewportIndex == 3 )
    {
        // MV
        Text::Print_ch( 'M', textState );
        Text::Print_ch( 'V', textState );

        float2 viewportUvPrevExpected = Geometry::GetScreenUv( gWorldToClipPrev, X );

        float2 viewportUvPrev = viewportUv + mv.xy;
        if( gMvScale.w != 0.0 )
            viewportUvPrev = Geometry::GetScreenUv( gWorldToClipPrev, X + mv );

        float2 uvDelta = ( viewportUvPrev - viewportUvPrevExpected ) * gRectSize;

        result.xyz = IsInScreenNearest( viewportUvPrev ) ? float3( abs( uvDelta ), 0 ) : float3( 0, 0, 1 );
        result.w = 1.0;
    }
    else if( viewportIndex == 0 )
    {
        // World units
        Text::Print_ch( 'U', textState );
        Text::Print_ch( 'N', textState );
        Text::Print_ch( 'I', textState );
        Text::Print_ch( 'T', textState );
        Text::Print_ch( 'S', textState );

        float roundingErrorCorrection = abs( viewZ ) * 0.001;

        result.xyz = frac( X + roundingErrorCorrection ) * float( !isInf );

        result.w = 1.0;
    }
    else if( viewportIndex == 8 )
    {
        // Diff-spec frames
        Text::Print_ch( 'D', textState );
        Text::Print_ch( 'I', textState );
        Text::Print_ch( 'F', textState );
        Text::Print_ch( 'F', textState );
        Text::Print_ch( '-', textState );
        Text::Print_ch( 'S', textState );
        Text::Print_ch( 'P', textState );
        Text::Print_ch( 'E', textState );
        Text::Print_ch( 'C', textState );
        Text::Print_ch( ' ', textState );
        Text::Print_ch( 'F', textState );
        Text::Print_ch( 'R', textState );
        Text::Print_ch( 'A', textState );
        Text::Print_ch( 'M', textState );
        Text::Print_ch( 'E', textState );
        Text::Print_ch( 'S', textState );

        float f = 1.0 - saturate( historyLength / max( max( gDiffMaxAccumulatedFrameNum, gSpecMaxAccumulatedFrameNum ), 1.0 ) );
        f = checkerboard && historyLength < 2.0 ? 0.75 : f;

        result.xyz = Color::ColorizeZucconi( viewportUv.y > 0.95 ? 1.0 - viewportUv.x : f * float( !isInf ) );
        result.w = 1.0;
    }
    else
        result = 0;

    if( Text::IsForeground( textState ) )
    {
        // Text
        float lum = Color::Luminance( result.xyz );
        result.xyz = lerp( 0.0, 1.0 - result.xyz, saturate( abs( lum - 0.5 ) / 0.25 ) ) ;
    }

    // Output
    gOut_Validation[ pixelPos ] = result;
}
