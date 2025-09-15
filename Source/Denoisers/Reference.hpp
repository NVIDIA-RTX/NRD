/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "../Shaders/REFERENCE_Copy.resources.hlsli"
#include "../Shaders/REFERENCE_TemporalAccumulation.resources.hlsli"

#define DENOISER_NAME Reference

void nrd::InstanceImpl::Add_Reference(DenoiserData& denoiserData) {
    denoiserData.settings.reference = ReferenceSettings();
    denoiserData.settingsSize = sizeof(denoiserData.settings.reference);

    enum class Permanent {
        HISTORY = PERMANENT_POOL_START,
    };

    AddTextureToPermanentPool({Format::RGBA32_SFLOAT, 1});

    std::array<ShaderMake::ShaderConstant, 0> commonDefines = {};

    PushPass("Temporal accumulation");
    {
        // Inputs
        PushInput(AsUint(ResourceType::IN_SIGNAL));

        // Outputs
        PushOutput(AsUint(Permanent::HISTORY));

        // Shaders
        AddDispatch(REFERENCE_TemporalAccumulation, commonDefines);
    }

    PushPass("Copy");
    {
        // Inputs
        PushInput(AsUint(Permanent::HISTORY));

        // Outputs
        PushOutput(AsUint(ResourceType::OUT_SIGNAL));

        // Shaders
        AddDispatch(REFERENCE_Copy, commonDefines);
    }
}

#undef DENOISER_NAME

void nrd::InstanceImpl::Update_Reference(const DenoiserData& denoiserData) {
    enum class Dispatch {
        ACCUMULATE,
        COPY,
    };

    const ReferenceSettings& settings = denoiserData.settings.reference;

    if (m_WorldToClip != m_WorldToClipPrev || m_CommonSettings.accumulationMode != AccumulationMode::CONTINUE || m_CommonSettings.rectSize[0] != m_CommonSettings.rectSizePrev[0] || m_CommonSettings.rectSize[1] != m_CommonSettings.rectSizePrev[1])
        m_AccumulatedFrameNum = 0;
    else {
        uint32_t maxAccumulatedFRameNum = min(settings.maxAccumulatedFrameNum, REFERENCE_MAX_HISTORY_FRAME_NUM);
        m_AccumulatedFrameNum = min(m_AccumulatedFrameNum + 1, maxAccumulatedFRameNum);
    }

    NRD_DECLARE_DIMS;

    { // ACCUMULATE
        REFERENCE_TemporalAccumulationConstants* consts = (REFERENCE_TemporalAccumulationConstants*)PushDispatch(denoiserData, AsUint(Dispatch::ACCUMULATE));
        consts->gAccumSpeed = 1.0f / (1.0f + m_AccumulatedFrameNum);
        consts->gDebug = m_CommonSettings.debug;
    }

    { // COPY
        REFERENCE_CopyConstants* consts = (REFERENCE_CopyConstants*)PushDispatch(denoiserData, AsUint(Dispatch::COPY));
        consts->gRectSizeInv = float2(1.0f / float(rectW), 1.0f / float(rectH));
        consts->gSplitScreen = m_CommonSettings.splitScreen;
    }
}
