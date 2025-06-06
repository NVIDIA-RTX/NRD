/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#define DENOISER_NAME REBLUR_DiffuseOcclusion
#define DIFF_TEMP1    AsUint(ResourceType::OUT_DIFF_HITDIST)
#define DIFF_TEMP2    AsUint(Transient::DIFF_TMP2)

void nrd::InstanceImpl::Add_ReblurDiffuseOcclusion(DenoiserData& denoiserData) {
    denoiserData.settings.reblur = ReblurSettings();
    denoiserData.settingsSize = sizeof(denoiserData.settings.reblur);

    enum class Permanent {
        PREV_VIEWZ = PERMANENT_POOL_START,
        PREV_NORMAL_ROUGHNESS,
        PREV_INTERNAL_DATA,
        DIFF_HISTORY,
        DIFF_FAST_HISTORY,
    };

    AddTextureToPermanentPool({REBLUR_FORMAT_PREV_VIEWZ, 1});
    AddTextureToPermanentPool({REBLUR_FORMAT_PREV_NORMAL_ROUGHNESS, 1});
    AddTextureToPermanentPool({REBLUR_FORMAT_PREV_INTERNAL_DATA, 1});
    AddTextureToPermanentPool({REBLUR_FORMAT_OCCLUSION, 1});
    AddTextureToPermanentPool({REBLUR_FORMAT_OCCLUSION_FAST_HISTORY, 1});

    enum class Transient {
        DATA1 = TRANSIENT_POOL_START,
        DIFF_TMP2,
        DIFF_FAST_HISTORY,
        TILES,
    };

    AddTextureToTransientPool({Format::R8_UNORM, 1});
    AddTextureToTransientPool({REBLUR_FORMAT_OCCLUSION, 1});
    AddTextureToTransientPool({REBLUR_FORMAT_OCCLUSION_FAST_HISTORY, 1});
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

    for (int i = 0; i < REBLUR_OCCLUSION_HITDIST_RECONSTRUCTION_PERMUTATION_NUM; i++) {
        bool is5x5 = (((i >> 0) & 0x1) != 0);

        PushPass("Hit distance reconstruction");
        {
            // Inputs
            PushInput(AsUint(Transient::TILES));
            PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
            PushInput(AsUint(ResourceType::IN_VIEWZ));
            PushInput(AsUint(ResourceType::IN_DIFF_HITDIST));

            // Outputs
            PushOutput(DIFF_TEMP1);

            // Shaders
            if (is5x5)
                AddDispatch(REBLUR_DiffuseOcclusion_HitDistReconstruction_5x5, REBLUR_HitDistReconstruction, 1);
            else
                AddDispatch(REBLUR_DiffuseOcclusion_HitDistReconstruction, REBLUR_HitDistReconstruction, 1);
        }
    }

    for (int i = 0; i < REBLUR_OCCLUSION_TEMPORAL_ACCUMULATION_PERMUTATION_NUM; i++) {
        bool hasDisocclusionThresholdMix = (((i >> 2) & 0x1) != 0);
        bool hasConfidenceInputs = (((i >> 1) & 0x1) != 0);
        bool isAfterReconstruction = (((i >> 0) & 0x1) != 0);

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
            PushInput(isAfterReconstruction ? DIFF_TEMP1 : AsUint(ResourceType::IN_DIFF_HITDIST));
            PushInput(AsUint(Permanent::DIFF_HISTORY));
            PushInput(AsUint(Permanent::DIFF_FAST_HISTORY));

            // Outputs
            PushOutput(DIFF_TEMP2);
            PushOutput(AsUint(Transient::DIFF_FAST_HISTORY));
            PushOutput(AsUint(Transient::DATA1));

            // Shaders
            AddDispatch(REBLUR_DiffuseOcclusion_TemporalAccumulation, REBLUR_TemporalAccumulation, 1);
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
        AddDispatch(REBLUR_DiffuseOcclusion_HistoryFix, REBLUR_HistoryFix, 1);
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
        AddDispatch(REBLUR_DiffuseOcclusion_Blur, REBLUR_Blur, 1);
    }

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
        PushOutput(AsUint(Permanent::PREV_INTERNAL_DATA));
        PushOutput(AsUint(ResourceType::OUT_DIFF_HITDIST));

        // Shaders
        AddDispatch(REBLUR_DiffuseOcclusion_PostBlur_NoTemporalStabilization, REBLUR_PostBlur, 1);
    }

    PushPass("Split screen");
    {
        // Inputs
        PushInput(AsUint(ResourceType::IN_VIEWZ));
        PushInput(AsUint(ResourceType::IN_DIFF_HITDIST));

        // Outputs
        PushOutput(AsUint(ResourceType::OUT_DIFF_HITDIST));

        // Shaders
        AddDispatch(REBLUR_Diffuse_SplitScreen, REBLUR_SplitScreen, 1);
    }

    REBLUR_ADD_VALIDATION_DISPATCH(Transient::DATA1, ResourceType::IN_DIFF_HITDIST, ResourceType::IN_DIFF_HITDIST);
}

#undef DENOISER_NAME
#undef DIFF_TEMP1
#undef DIFF_TEMP2
