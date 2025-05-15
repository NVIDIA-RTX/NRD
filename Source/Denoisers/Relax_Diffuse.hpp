/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#define DENOISER_NAME RELAX_Diffuse

void nrd::InstanceImpl::Add_RelaxDiffuse(DenoiserData& denoiserData) {
    denoiserData.settings.relax = RelaxSettings();
    denoiserData.settingsSize = sizeof(denoiserData.settings.relax);

    enum class Permanent {
        DIFF_ILLUM_PREV = PERMANENT_POOL_START,
        DIFF_ILLUM_RESPONSIVE_PREV,
        HISTORY_LENGTH_PREV,
        NORMAL_ROUGHNESS_PREV,
        MATERIAL_ID_PREV,
        VIEWZ_PREV
    };

    AddTextureToPermanentPool({Format::RGBA16_SFLOAT, 1});
    AddTextureToPermanentPool({Format::RGBA16_SFLOAT, 1});
    AddTextureToPermanentPool({Format::R8_UNORM, 1});
    AddTextureToPermanentPool({Format::RGBA8_UNORM, 1});
    AddTextureToPermanentPool({Format::R8_UNORM, 1});
    AddTextureToPermanentPool({Format::R32_SFLOAT, 1});

    enum class Transient {
        DIFF_ILLUM_PING = TRANSIENT_POOL_START,
        DIFF_ILLUM_PONG,
        TILES,
        HISTORY_LENGTH
    };

    AddTextureToTransientPool({Format::RGBA16_SFLOAT, 1});
    AddTextureToTransientPool({Format::RGBA16_SFLOAT, 1});
    AddTextureToTransientPool({Format::R8_UNORM, 16});
    AddTextureToTransientPool({Format::R8_UNORM, 1});

    PushPass("Classify tiles");
    {
        // Inputs
        PushInput(AsUint(ResourceType::IN_VIEWZ));

        // Outputs
        PushOutput(AsUint(Transient::TILES));

        // Shaders
        AddDispatch(RELAX_ClassifyTiles, RELAX_ClassifyTiles, 1);
    }

    for (int i = 0; i < RELAX_HITDIST_RECONSTRUCTION_PERMUTATION_NUM; i++) {
        bool is5x5 = (((i >> 0) & 0x1) != 0);

        PushPass("Hit distance reconstruction");
        {
            // Inputs
            PushInput(AsUint(Transient::TILES));
            PushInput(AsUint(ResourceType::IN_DIFF_RADIANCE_HITDIST));
            PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
            PushInput(AsUint(ResourceType::IN_VIEWZ));

            // Outputs
            PushOutput(AsUint(Transient::DIFF_ILLUM_PING));

            // Shaders
            if (is5x5)
                AddDispatch(RELAX_Diffuse_HitDistReconstruction_5x5, RELAX_HitDistReconstruction, 1);
            else
                AddDispatch(RELAX_Diffuse_HitDistReconstruction, RELAX_HitDistReconstruction, 1);
        }
    }

    for (int i = 0; i < RELAX_PREPASS_PERMUTATION_NUM; i++) {
        bool isAfterReconstruction = (((i >> 0) & 0x1) != 0);

        PushPass("Pre-pass");
        {
            // Inputs
            PushInput(AsUint(Transient::TILES));
            PushInput(isAfterReconstruction ? AsUint(Transient::DIFF_ILLUM_PING) : AsUint(ResourceType::IN_DIFF_RADIANCE_HITDIST));
            PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
            PushInput(AsUint(ResourceType::IN_VIEWZ));

            // Outputs
            PushOutput(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));

            // Shaders
            AddDispatch(RELAX_Diffuse_PrePass, RELAX_PrePass, 1);
        }
    }

    for (int i = 0; i < RELAX_TEMPORAL_ACCUMULATION_PERMUTATION_NUM; i++) {
        bool hasDisocclusionThresholdMix = (((i >> 1) & 0x1) != 0);
        bool hasConfidenceInputs = (((i >> 0) & 0x1) != 0);

        PushPass("Temporal accumulation");
        {
            // Inputs
            PushInput(AsUint(Transient::TILES));
            PushInput(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));
            PushInput(AsUint(ResourceType::IN_MV));
            PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
            PushInput(AsUint(ResourceType::IN_VIEWZ));
            PushInput(AsUint(Permanent::DIFF_ILLUM_RESPONSIVE_PREV));
            PushInput(AsUint(Permanent::DIFF_ILLUM_PREV));
            PushInput(AsUint(Permanent::NORMAL_ROUGHNESS_PREV));
            PushInput(AsUint(Permanent::VIEWZ_PREV));
            PushInput(AsUint(Permanent::HISTORY_LENGTH_PREV));
            PushInput(AsUint(Permanent::MATERIAL_ID_PREV));
            PushInput(hasConfidenceInputs ? AsUint(ResourceType::IN_DIFF_CONFIDENCE) : RELAX_DUMMY);
            PushInput(hasDisocclusionThresholdMix ? AsUint(ResourceType::IN_DISOCCLUSION_THRESHOLD_MIX) : RELAX_DUMMY);

            // Outputs
            PushOutput(AsUint(Transient::DIFF_ILLUM_PING));
            PushOutput(AsUint(Transient::DIFF_ILLUM_PONG));
            PushOutput(AsUint(Transient::HISTORY_LENGTH));

            // Shaders
            AddDispatch(RELAX_Diffuse_TemporalAccumulation, RELAX_TemporalAccumulation, 1);
        }
    }

    PushPass("History fix");
    {
        // Inputs
        PushInput(AsUint(Transient::TILES));
        PushInput(AsUint(Transient::DIFF_ILLUM_PING));
        PushInput(AsUint(Transient::HISTORY_LENGTH));
        PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
        PushInput(AsUint(ResourceType::IN_VIEWZ));

        // Outputs
        PushOutput(AsUint(Transient::DIFF_ILLUM_PONG));

        // Shaders
        AddDispatch(RELAX_Diffuse_HistoryFix, RELAX_HistoryFix, 1);
    }

    PushPass("History clamping");
    {
        // Inputs
        PushInput(AsUint(Transient::TILES));
        PushInput(AsUint(ResourceType::IN_VIEWZ));
        PushInput(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));
        PushInput(AsUint(Transient::DIFF_ILLUM_PING));
        PushInput(AsUint(Transient::DIFF_ILLUM_PONG));
        PushInput(AsUint(Transient::HISTORY_LENGTH));

        // Outputs
        PushOutput(AsUint(Permanent::DIFF_ILLUM_PREV));
        PushOutput(AsUint(Permanent::DIFF_ILLUM_RESPONSIVE_PREV));
        PushOutput(AsUint(Permanent::HISTORY_LENGTH_PREV));

        // Shaders
        AddDispatch(RELAX_Diffuse_HistoryClamping, RELAX_HistoryClamping, 1);
    }

    PushPass("Copy");
    {
        // Inputs
        PushInput(AsUint(Permanent::DIFF_ILLUM_PREV));

        // Outputs
        PushOutput(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));

        // Shaders
        AddDispatch(RELAX_Diffuse_Copy, RELAX_Copy, 1);
    }

    PushPass("Anti-firefly");
    {
        // Inputs
        PushInput(AsUint(Transient::TILES));
        PushInput(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));
        PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
        PushInput(AsUint(ResourceType::IN_VIEWZ));

        // Outputs
        PushOutput(AsUint(Permanent::DIFF_ILLUM_PREV));

        // Shaders
        AddDispatch(RELAX_Diffuse_AntiFirefly, RELAX_AntiFirefly, 1);
    }

    for (int i = 0; i < RELAX_ATROUS_PERMUTATION_NUM; i++) {
        bool hasConfidenceInputs = (((i >> 0) & 0x1) != 0);

        for (int j = 0; j < RELAX_ATROUS_BINDING_VARIANT_NUM; j++) {
            bool isSmem = j == 0;
            bool isEven = j % 2 == 0;
            bool isLast = j > 2;

            if (isSmem)
                PushPass("A-trous (SMEM)");
            else
                PushPass("A-trous");

            {
                // Inputs
                PushInput(AsUint(Transient::TILES));

                if (isSmem)
                    PushInput(AsUint(Permanent::DIFF_ILLUM_PREV));
                else
                    PushInput(isEven ? AsUint(Transient::DIFF_ILLUM_PONG) : AsUint(Transient::DIFF_ILLUM_PING));

                PushInput(AsUint(Transient::HISTORY_LENGTH));
                PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
                PushInput(AsUint(ResourceType::IN_VIEWZ));
                PushInput(hasConfidenceInputs ? AsUint(ResourceType::IN_DIFF_CONFIDENCE) : RELAX_DUMMY);

                // Outputs
                if (isLast)
                    PushOutput(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));
                else
                    PushOutput(isEven ? AsUint(Transient::DIFF_ILLUM_PING) : AsUint(Transient::DIFF_ILLUM_PONG));

                if (isSmem) {
                    PushOutput(AsUint(Permanent::NORMAL_ROUGHNESS_PREV));
                    PushOutput(AsUint(Permanent::MATERIAL_ID_PREV));
                    PushOutput(AsUint(Permanent::VIEWZ_PREV));
                }

                // Shaders
                constexpr uint32_t maxRepeatNum = (RELAX_MAX_ATROUS_PASS_NUM - 2 + 1) / 2;
                if (isSmem)
                    AddDispatch(RELAX_Diffuse_AtrousSmem, RELAX_AtrousSmem, 1);
                else
                    AddDispatchRepeated(RELAX_Diffuse_Atrous, RELAX_Atrous, 1, maxRepeatNum);
            }
        }
    }

    PushPass("Split screen");
    {
        // Inputs
        PushInput(AsUint(ResourceType::IN_VIEWZ));
        PushInput(AsUint(ResourceType::IN_DIFF_RADIANCE_HITDIST));

        // Outputs
        PushOutput(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));

        // Shaders
        AddDispatch(RELAX_Diffuse_SplitScreen, RELAX_SplitScreen, 1);
    }

    RELAX_ADD_VALIDATION_DISPATCH;
}

#undef DENOISER_NAME
