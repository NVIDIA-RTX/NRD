/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "InstanceImpl.h"

// Shaders
#if NRD_EMBEDS_DXBC_SHADERS
#    include "REFERENCE_Copy.cs.dxbc.h"
#    include "REFERENCE_TemporalAccumulation.cs.dxbc.h"
#endif

#if NRD_EMBEDS_DXIL_SHADERS
#    include "REFERENCE_Copy.cs.dxil.h"
#    include "REFERENCE_TemporalAccumulation.cs.dxil.h"
#endif

#if NRD_EMBEDS_SPIRV_SHADERS
#    include "REFERENCE_Copy.cs.spirv.h"
#    include "REFERENCE_TemporalAccumulation.cs.spirv.h"
#endif

// Denoisers
#include "Denoisers/Reference.hpp"
