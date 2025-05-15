/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#define DENOISER_NAME REBLUR_DirectionalOcclusion
#define DIFF_TEMP1    AsUint(ResourceType::OUT_DIFF_DIRECTION_HITDIST)
#define DIFF_TEMP2    AsUint(Transient::DIFF_TMP2)

void nrd::InstanceImpl::Add_ReblurDiffuseDirectionalOcclusion(DenoiserData& denoiserData) {
    denoiserData.settings.reblur = ReblurSettings();
    denoiserData.settingsSize = sizeof(denoiserData.settings.reblur);

    // IMPORTANT: uses SNORM / UNORM 16-bit textures to maximize bits utilization and uniformity

    enum class Permanent {
        PREV_VIEWZ = PERMANENT_POOL_START,
        PREV_NORMAL_ROUGHNESS,
        PREV_INTERNAL_DATA,
        DIFF_HISTORY,
        DIFF_FAST_HISTORY,
        DIFF_HISTORY_STABILIZED_PING,
        DIFF_HISTORY_STABILIZED_PONG,
    };

    AddTextureToPermanentPool({REBLUR_FORMAT_PREV_VIEWZ, 1});
    AddTextureToPermanentPool({REBLUR_FORMAT_PREV_NORMAL_ROUGHNESS, 1});
    AddTextureToPermanentPool({REBLUR_FORMAT_PREV_INTERNAL_DATA, 1});
    AddTextureToPermanentPool({REBLUR_FORMAT_DIRECTIONAL_OCCLUSION, 1});
    AddTextureToPermanentPool({REBLUR_FORMAT_DIRECTIONAL_OCCLUSION_FAST_HISTORY, 1});
    AddTextureToPermanentPool({Format::R16_SFLOAT, 1});
    AddTextureToPermanentPool({Format::R16_SFLOAT, 1});

    enum class Transient {
        DATA1 = TRANSIENT_POOL_START,
        DATA2,
        DIFF_TMP2,
        DIFF_FAST_HISTORY,
        TILES,
    };

    AddTextureToTransientPool({Format::R8_UNORM, 1});
    AddTextureToTransientPool({Format::R8_UINT, 1});
    AddTextureToTransientPool({REBLUR_FORMAT_DIRECTIONAL_OCCLUSION, 1});
    AddTextureToTransientPool({REBLUR_FORMAT_DIRECTIONAL_OCCLUSION_FAST_HISTORY, 1});
    AddTextureToTransientPool({REBLUR_FORMAT_TILES, 16});

    PushPass("Classify tiles");
    {
        // Inputs
        PushInput(AsUint(ResourceType::IN_VIEWZ));

        // Outputs
        PushOutput(AsUint(Transient::TILES));

        // Shaders
        AddDispatch(REBLUR_ClassifyTiles, REBLUR_ClassifyTiles, 1);
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
            PushInput(AsUint(ResourceType::IN_DIFF_DIRECTION_HITDIST));

            // Outputs
            PushOutput(isPrepassEnabled ? DIFF_TEMP2 : DIFF_TEMP1);

            // Shaders
            if (is5x5) {
                AddDispatch(REBLUR_Diffuse_HitDistReconstruction_5x5, REBLUR_HitDistReconstruction, 1);
                AddDispatch(REBLUR_Perf_Diffuse_HitDistReconstruction_5x5, REBLUR_HitDistReconstruction, 1);
            } else {
                AddDispatch(REBLUR_Diffuse_HitDistReconstruction, REBLUR_HitDistReconstruction, 1);
                AddDispatch(REBLUR_Perf_Diffuse_HitDistReconstruction, REBLUR_HitDistReconstruction, 1);
            }
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
            PushInput(isAfterReconstruction ? DIFF_TEMP2 : AsUint(ResourceType::IN_DIFF_DIRECTION_HITDIST));

            // Outputs
            PushOutput(DIFF_TEMP1);

            // Shaders
            AddDispatch(REBLUR_DiffuseDirectionalOcclusion_PrePass, REBLUR_PrePass, 1);
            AddDispatch(REBLUR_Perf_DiffuseDirectionalOcclusion_PrePass, REBLUR_PrePass, 1);
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
            PushInput(isAfterPrepass ? DIFF_TEMP1 : AsUint(ResourceType::IN_DIFF_DIRECTION_HITDIST));
            PushInput(AsUint(Permanent::DIFF_HISTORY));
            PushInput(AsUint(Permanent::DIFF_FAST_HISTORY));

            // Outputs
            PushOutput(DIFF_TEMP2);
            PushOutput(AsUint(Transient::DIFF_FAST_HISTORY));
            PushOutput(AsUint(Transient::DATA1));
            PushOutput(AsUint(Transient::DATA2));

            // Shaders
            AddDispatch(REBLUR_DiffuseDirectionalOcclusion_TemporalAccumulation, REBLUR_TemporalAccumulation, 1);
            AddDispatch(REBLUR_Perf_DiffuseDirectionalOcclusion_TemporalAccumulation, REBLUR_TemporalAccumulation, 1);
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
        PushInput(AsUint(Transient::DIFF_FAST_HISTORY));

        // Outputs
        PushOutput(DIFF_TEMP1);
        PushOutput(AsUint(Permanent::DIFF_FAST_HISTORY));

        // Shaders
        AddDispatch(REBLUR_DiffuseDirectionalOcclusion_HistoryFix, REBLUR_HistoryFix, 1);
        AddDispatch(REBLUR_Perf_DiffuseDirectionalOcclusion_HistoryFix, REBLUR_HistoryFix, 1);
    }

    PushPass("Blur");
    {
        // Inputs
        PushInput(AsUint(Transient::TILES));
        PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
        PushInput(AsUint(Transient::DATA1));
        PushInput(DIFF_TEMP1);
        PushInput(AsUint(ResourceType::IN_VIEWZ));

        // Outputs
        PushOutput(DIFF_TEMP2);
        PushOutput(AsUint(Permanent::PREV_VIEWZ));

        // Shaders
        AddDispatch(REBLUR_DiffuseDirectionalOcclusion_Blur, REBLUR_Blur, 1);
        AddDispatch(REBLUR_Perf_DiffuseDirectionalOcclusion_Blur, REBLUR_Blur, 1);
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
            PushInput(AsUint(Permanent::PREV_VIEWZ));

            // Outputs
            PushOutput(AsUint(Permanent::PREV_NORMAL_ROUGHNESS));
            PushOutput(AsUint(Permanent::DIFF_HISTORY));

            if (!isTemporalStabilization) {
                PushOutput(AsUint(Permanent::PREV_INTERNAL_DATA));
                PushOutput(AsUint(ResourceType::OUT_DIFF_DIRECTION_HITDIST));
            }

            // Shaders
            if (isTemporalStabilization) {
                AddDispatch(REBLUR_DiffuseDirectionalOcclusion_PostBlur, REBLUR_PostBlur, 1);
                AddDispatch(REBLUR_Perf_DiffuseDirectionalOcclusion_PostBlur, REBLUR_PostBlur, 1);
            } else {
                AddDispatch(REBLUR_DiffuseDirectionalOcclusion_PostBlur_NoTemporalStabilization, REBLUR_PostBlur, 1);
                AddDispatch(REBLUR_Perf_DiffuseDirectionalOcclusion_PostBlur_NoTemporalStabilization, REBLUR_PostBlur, 1);
            }
        }
    }

    for (int i = 0; i < REBLUR_TEMPORAL_STABILIZATION_PERMUTATION_NUM; i++) {
        PushPass("Temporal stabilization");
        {
            // Inputs
            PushInput(AsUint(Transient::TILES));
            PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
            PushInput(AsUint(Permanent::PREV_VIEWZ));
            PushInput(AsUint(Transient::DATA1));
            PushInput(AsUint(Transient::DATA2));
            PushInput(AsUint(Permanent::DIFF_HISTORY));
            PushInput(AsUint(Permanent::DIFF_HISTORY_STABILIZED_PING), AsUint(Permanent::DIFF_HISTORY_STABILIZED_PONG));

            // Outputs
            PushOutput(AsUint(ResourceType::IN_MV));
            PushOutput(AsUint(Permanent::PREV_INTERNAL_DATA));
            PushOutput(AsUint(ResourceType::OUT_DIFF_DIRECTION_HITDIST));
            PushOutput(AsUint(Permanent::DIFF_HISTORY_STABILIZED_PONG), AsUint(Permanent::DIFF_HISTORY_STABILIZED_PING));

            // Shaders
            AddDispatch(REBLUR_DiffuseDirectionalOcclusion_TemporalStabilization, REBLUR_TemporalStabilization, 1);
            AddDispatch(REBLUR_Perf_DiffuseDirectionalOcclusion_TemporalStabilization, REBLUR_TemporalStabilization, 1);
        }
    }

    PushPass("Split screen");
    {
        // Inputs
        PushInput(AsUint(ResourceType::IN_VIEWZ));
        PushInput(AsUint(ResourceType::IN_DIFF_DIRECTION_HITDIST));

        // Outputs
        PushOutput(AsUint(ResourceType::OUT_DIFF_DIRECTION_HITDIST));

        // Shaders
        AddDispatch(REBLUR_Diffuse_SplitScreen, REBLUR_SplitScreen, 1);
    }

    REBLUR_ADD_VALIDATION_DISPATCH(Transient::DATA2, ResourceType::IN_DIFF_DIRECTION_HITDIST, ResourceType::IN_DIFF_DIRECTION_HITDIST);
}

#undef DENOISER_NAME
#undef DIFF_TEMP1
#undef DIFF_TEMP2
