/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

NRD_CONSTANTS_START( RELAX_ClassifyTilesConstants )
    RELAX_SHARED_CONSTANTS
NRD_CONSTANTS_END

NRD_INPUTS_START
    NRD_INPUT( Texture2D<float>, gIn_ViewZ, t, 0 )
NRD_INPUTS_END

NRD_OUTPUTS_START
    NRD_OUTPUT( RWTexture2D<float>, gOut_Tiles, u, 0 )
NRD_OUTPUTS_END

// Macro magic
#define RELAX_ClassifyTilesGroupX 16
#define RELAX_ClassifyTilesGroupY 16

// Redirection
#undef GROUP_X
#undef GROUP_Y
#define GROUP_X RELAX_ClassifyTilesGroupX
#define GROUP_Y RELAX_ClassifyTilesGroupY
