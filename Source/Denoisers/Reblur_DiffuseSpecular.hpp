/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#define DENOISER_NAME REBLUR_DiffuseSpecular
#define DIFF_TEMP1    AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST)
#define DIFF_TEMP2    AsUint(Transient::DIFF_TMP2)
#define SPEC_TEMP1    AsUint(ResourceType::OUT_SPEC_RADIANCE_HITDIST)
#define SPEC_TEMP2    AsUint(Transient::SPEC_TMP2)

void nrd::InstanceImpl::Add_ReblurDiffuseSpecular(DenoiserData& denoiserData) {
    denoiserData.settings.reblur = ReblurSettings();
    denoiserData.settingsSize = sizeof(denoiserData.settings.reblur);

    enum class Permanent {
        PREV_VIEWZ = PERMANENT_POOL_START,
        PREV_NORMAL_ROUGHNESS,
        PREV_INTERNAL_DATA,
        DIFF_HISTORY,
        DIFF_FAST_HISTORY,
        DIFF_HISTORY_STABILIZED_PING,
        DIFF_HISTORY_STABILIZED_PONG,
        SPEC_HISTORY,
        SPEC_FAST_HISTORY,
        SPEC_HISTORY_STABILIZED_PING,
        SPEC_HISTORY_STABILIZED_PONG,
        SPEC_HITDIST_FOR_TRACKING_PING,
        SPEC_HITDIST_FOR_TRACKING_PONG,
    };

    AddTextureToPermanentPool({REBLUR_FORMAT_PREV_VIEWZ, 1});
    AddTextureToPermanentPool({REBLUR_FORMAT_PREV_NORMAL_ROUGHNESS, 1});
    AddTextureToPermanentPool({REBLUR_FORMAT_PREV_INTERNAL_DATA, 1});
    AddTextureToPermanentPool({REBLUR_FORMAT, 1});
    AddTextureToPermanentPool({REBLUR_FORMAT_FAST_HISTORY, 1});
    AddTextureToPermanentPool({Format::R16_SFLOAT, 1});
    AddTextureToPermanentPool({Format::R16_SFLOAT, 1});
    AddTextureToPermanentPool({REBLUR_FORMAT, 1});
    AddTextureToPermanentPool({REBLUR_FORMAT_FAST_HISTORY, 1});
    AddTextureToPermanentPool({Format::R16_SFLOAT, 1});
    AddTextureToPermanentPool({Format::R16_SFLOAT, 1});
    AddTextureToPermanentPool({REBLUR_FORMAT_HITDIST_FOR_TRACKING, 1});
    AddTextureToPermanentPool({REBLUR_FORMAT_HITDIST_FOR_TRACKING, 1});

    enum class Transient {
        DATA1 = TRANSIENT_POOL_START,
        DATA2,
        SPEC_HITDIST_FOR_TRACKING,
        DIFF_TMP2,
        DIFF_FAST_HISTORY,
        SPEC_TMP2,
        SPEC_FAST_HISTORY,
        TILES,
    };

    AddTextureToTransientPool({Format::RG8_UNORM, 1});
    AddTextureToTransientPool({Format::R32_UINT, 1});
    AddTextureToTransientPool({REBLUR_FORMAT_HITDIST_FOR_TRACKING, 1});
    AddTextureToTransientPool({REBLUR_FORMAT, 1});
    AddTextureToTransientPool({REBLUR_FORMAT_FAST_HISTORY, 1});
    AddTextureToTransientPool({REBLUR_FORMAT, 1});
    AddTextureToTransientPool({REBLUR_FORMAT_FAST_HISTORY, 1});
    AddTextureToTransientPool({REBLUR_FORMAT_TILES, 16});

    std::array<ShaderMake::ShaderConstant, 2> commonDefines = {{
        {"NRD_SIGNAL", NRD_DIFFUSE_SPECULAR},
        {"NRD_MODE", NRD_RADIANCE},
    }};

    PushPass("Classify tiles");
    {
        // Inputs
        PushInput(AsUint(ResourceType::IN_VIEWZ));

        // Outputs
        PushOutput(AsUint(Transient::TILES));

        // Shaders
        std::array<ShaderMake::ShaderConstant, 0> defines = {};
        AddDispatch(REBLUR_ClassifyTiles, defines);
    }

    for (int i = 0; i < REBLUR_HITDIST_RECONSTRUCTION_PERMUTATION_NUM; i++) {
        bool is5x5 = (((i >> 1) & 0x1) != 0);
        bool isPrepassEnabled = (((i >> 0) & 0x1) != 0);

        PushPass("Hit distance reconstruction");
        {
            // Inputs
            PushInput(AsUint(Transient::TILES));
            PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
            PushInput(AsUint(ResourceType::IN_VIEWZ));
            PushInput(AsUint(ResourceType::IN_DIFF_RADIANCE_HITDIST));
            PushInput(AsUint(ResourceType::IN_SPEC_RADIANCE_HITDIST));

            // Outputs
            PushOutput(isPrepassEnabled ? DIFF_TEMP2 : DIFF_TEMP1);
            PushOutput(isPrepassEnabled ? SPEC_TEMP2 : SPEC_TEMP1);

            // Shaders
            std::array<ShaderMake::ShaderConstant, 3> defines = {{
                commonDefines[0],
                commonDefines[1],
                {"MODE_5X5", is5x5 ? "1" : "0"},
            }};
            AddDispatch(REBLUR_HitDistReconstruction, defines);
        }
    }

    for (int i = 0; i < REBLUR_PREPASS_PERMUTATION_NUM; i++) {
        bool isAfterReconstruction = (((i >> 0) & 0x1) != 0);

        PushPass("Pre-pass");
        {
            // Inputs
            PushInput(AsUint(Transient::TILES));
            PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
            PushInput(AsUint(ResourceType::IN_VIEWZ));
            PushInput(isAfterReconstruction ? DIFF_TEMP2 : AsUint(ResourceType::IN_DIFF_RADIANCE_HITDIST));
            PushInput(isAfterReconstruction ? SPEC_TEMP2 : AsUint(ResourceType::IN_SPEC_RADIANCE_HITDIST));

            // Outputs
            PushOutput(DIFF_TEMP1);
            PushOutput(SPEC_TEMP1);
            PushOutput(AsUint(Transient::SPEC_HITDIST_FOR_TRACKING));

            // Shaders
            AddDispatch(REBLUR_PrePass, commonDefines);
        }
    }

    for (int i = 0; i < REBLUR_TEMPORAL_ACCUMULATION_PERMUTATION_NUM; i++) {
        bool hasDisocclusionThresholdMix = (((i >> 2) & 0x1) != 0);
        bool hasConfidenceInputs = (((i >> 1) & 0x1) != 0);
        bool isAfterPrepass = (((i >> 0) & 0x1) != 0);

        PushPass("Temporal accumulation");
        {
            // Inputs
            PushInput(AsUint(Transient::TILES));
            PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
            PushInput(AsUint(ResourceType::IN_VIEWZ));
            PushInput(AsUint(ResourceType::IN_MV));
            PushInput(AsUint(Permanent::PREV_VIEWZ));
            PushInput(AsUint(Permanent::PREV_NORMAL_ROUGHNESS));
            PushInput(AsUint(Permanent::PREV_INTERNAL_DATA));
            PushInput(hasDisocclusionThresholdMix ? AsUint(ResourceType::IN_DISOCCLUSION_THRESHOLD_MIX) : REBLUR_DUMMY);
            PushInput(hasConfidenceInputs ? AsUint(ResourceType::IN_DIFF_CONFIDENCE) : REBLUR_DUMMY);
            PushInput(hasConfidenceInputs ? AsUint(ResourceType::IN_SPEC_CONFIDENCE) : REBLUR_DUMMY);
            PushInput(isAfterPrepass ? DIFF_TEMP1 : AsUint(ResourceType::IN_DIFF_RADIANCE_HITDIST));
            PushInput(isAfterPrepass ? SPEC_TEMP1 : AsUint(ResourceType::IN_SPEC_RADIANCE_HITDIST));
            PushInput(AsUint(Permanent::DIFF_HISTORY));
            PushInput(AsUint(Permanent::SPEC_HISTORY));
            PushInput(AsUint(Permanent::DIFF_FAST_HISTORY));
            PushInput(AsUint(Permanent::SPEC_FAST_HISTORY));
            PushInput(AsUint(Permanent::SPEC_HITDIST_FOR_TRACKING_PING), AsUint(Permanent::SPEC_HITDIST_FOR_TRACKING_PONG));
            PushInput(AsUint(Transient::SPEC_HITDIST_FOR_TRACKING));

            // Outputs
            PushOutput(DIFF_TEMP2);
            PushOutput(SPEC_TEMP2);
            PushOutput(AsUint(Transient::DIFF_FAST_HISTORY));
            PushOutput(AsUint(Transient::SPEC_FAST_HISTORY));
            PushOutput(AsUint(Permanent::SPEC_HITDIST_FOR_TRACKING_PONG), AsUint(Permanent::SPEC_HITDIST_FOR_TRACKING_PING));
            PushOutput(AsUint(Transient::DATA1));
            PushOutput(AsUint(Transient::DATA2));

            // Shaders
            AddDispatch(REBLUR_TemporalAccumulation, commonDefines);
        }
    }

    PushPass("History fix");
    {
        // Inputs
        PushInput(AsUint(Transient::TILES));
        PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
        PushInput(AsUint(Transient::DATA1));
        PushInput(AsUint(ResourceType::IN_VIEWZ));
        PushInput(DIFF_TEMP2);
        PushInput(SPEC_TEMP2);
        PushInput(AsUint(Transient::DIFF_FAST_HISTORY));
        PushInput(AsUint(Transient::SPEC_FAST_HISTORY));

        // Outputs
        PushOutput(DIFF_TEMP1);
        PushOutput(SPEC_TEMP1);
        PushOutput(AsUint(Permanent::DIFF_FAST_HISTORY));
        PushOutput(AsUint(Permanent::SPEC_FAST_HISTORY));

        // Shaders
        AddDispatch(REBLUR_HistoryFix, commonDefines);
    }

    PushPass("Blur");
    {
        // Inputs
        PushInput(AsUint(Transient::TILES));
        PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
        PushInput(AsUint(Transient::DATA1));
        PushInput(DIFF_TEMP1);
        PushInput(SPEC_TEMP1);
        PushInput(AsUint(ResourceType::IN_VIEWZ));

        // Outputs
        PushOutput(DIFF_TEMP2);
        PushOutput(SPEC_TEMP2);
        PushOutput(AsUint(Permanent::PREV_VIEWZ));

        // Shaders
        AddDispatch(REBLUR_Blur, commonDefines);
    }

    for (int i = 0; i < REBLUR_POST_BLUR_PERMUTATION_NUM; i++) {
        bool isTemporalStabilization = (((i >> 0) & 0x1) != 0);

        PushPass("Post-blur");
        {
            // Inputs
            PushInput(AsUint(Transient::TILES));
            PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
            PushInput(AsUint(Transient::DATA1));
            PushInput(DIFF_TEMP2);
            PushInput(SPEC_TEMP2);
            PushInput(AsUint(Permanent::PREV_VIEWZ));

            // Outputs
            PushOutput(AsUint(Permanent::PREV_NORMAL_ROUGHNESS));
            PushOutput(AsUint(Permanent::DIFF_HISTORY));
            PushOutput(AsUint(Permanent::SPEC_HISTORY));

            if (!isTemporalStabilization) {
                PushOutput(AsUint(Permanent::PREV_INTERNAL_DATA));
                PushOutput(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));
                PushOutput(AsUint(ResourceType::OUT_SPEC_RADIANCE_HITDIST));
            }

            // Shaders
            std::array<ShaderMake::ShaderConstant, 3> defines = {{
                commonDefines[0],
                commonDefines[1],
                {"TEMPORAL_STABILIZATION", isTemporalStabilization ? "1" : "0"},
            }};
            AddDispatch(REBLUR_PostBlur, defines);
        }
    }

    for (int i = 0; i < REBLUR_TEMPORAL_STABILIZATION_PERMUTATION_NUM; i++) {
        bool hasRf0AndMetalness = (((i >> 0) & 0x1) != 0);

        PushPass("Temporal stabilization");
        {
            // Inputs
            PushInput(AsUint(Transient::TILES));
            PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
            PushInput(hasRf0AndMetalness ? AsUint(ResourceType::IN_BASECOLOR_METALNESS) : REBLUR_DUMMY);
            PushInput(AsUint(Permanent::PREV_VIEWZ));
            PushInput(AsUint(Transient::DATA1));
            PushInput(AsUint(Transient::DATA2));
            PushInput(AsUint(Permanent::DIFF_HISTORY));
            PushInput(AsUint(Permanent::SPEC_HISTORY));
            PushInput(AsUint(Permanent::DIFF_HISTORY_STABILIZED_PING), AsUint(Permanent::DIFF_HISTORY_STABILIZED_PONG));
            PushInput(AsUint(Permanent::SPEC_HISTORY_STABILIZED_PING), AsUint(Permanent::SPEC_HISTORY_STABILIZED_PONG));
            PushInput(AsUint(Permanent::SPEC_HITDIST_FOR_TRACKING_PONG), AsUint(Permanent::SPEC_HITDIST_FOR_TRACKING_PING));

            // Outputs
            PushOutput(AsUint(ResourceType::IN_MV));
            PushOutput(AsUint(Permanent::PREV_INTERNAL_DATA));
            PushOutput(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));
            PushOutput(AsUint(ResourceType::OUT_SPEC_RADIANCE_HITDIST));
            PushOutput(AsUint(Permanent::DIFF_HISTORY_STABILIZED_PONG), AsUint(Permanent::DIFF_HISTORY_STABILIZED_PING));
            PushOutput(AsUint(Permanent::SPEC_HISTORY_STABILIZED_PONG), AsUint(Permanent::SPEC_HISTORY_STABILIZED_PING));

            // Shaders
            AddDispatch(REBLUR_TemporalStabilization, commonDefines);
        }
    }

    PushPass("Split screen");
    {
        // Inputs
        PushInput(AsUint(ResourceType::IN_VIEWZ));
        PushInput(AsUint(ResourceType::IN_DIFF_RADIANCE_HITDIST));
        PushInput(AsUint(ResourceType::IN_SPEC_RADIANCE_HITDIST));

        // Outputs
        PushOutput(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));
        PushOutput(AsUint(ResourceType::OUT_SPEC_RADIANCE_HITDIST));

        // Shaders
        AddDispatch(REBLUR_SplitScreen, commonDefines);
    }

    REBLUR_ADD_VALIDATION_DISPATCH(Transient::DATA2, ResourceType::IN_DIFF_RADIANCE_HITDIST, ResourceType::IN_SPEC_RADIANCE_HITDIST);
}

#undef DENOISER_NAME
#undef DIFF_TEMP1
#undef SPEC_TEMP1
#undef DIFF_TEMP2
#undef SPEC_TEMP2
