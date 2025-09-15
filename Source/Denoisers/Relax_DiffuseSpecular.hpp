/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#define DENOISER_NAME RELAX_DiffuseSpecular

void nrd::InstanceImpl::Add_RelaxDiffuseSpecular(DenoiserData& denoiserData) {
    denoiserData.settings.relax = RelaxSettings();
    denoiserData.settingsSize = sizeof(denoiserData.settings.relax);

    enum class Permanent {
        SPEC_ILLUM_PREV = PERMANENT_POOL_START,
        DIFF_ILLUM_PREV,
        SPEC_ILLUM_RESPONSIVE_PREV,
        DIFF_ILLUM_RESPONSIVE_PREV,
        REFLECTION_HIT_T_CURR,
        REFLECTION_HIT_T_PREV,
        HISTORY_LENGTH_PREV,
        NORMAL_ROUGHNESS_PREV,
        MATERIAL_ID_PREV,
        VIEWZ_PREV,
    };

    AddTextureToPermanentPool({Format::RGBA16_SFLOAT, 1});
    AddTextureToPermanentPool({Format::RGBA16_SFLOAT, 1});
    AddTextureToPermanentPool({Format::RGBA16_SFLOAT, 1});
    AddTextureToPermanentPool({Format::RGBA16_SFLOAT, 1});
    AddTextureToPermanentPool({Format::R16_SFLOAT, 1});
    AddTextureToPermanentPool({Format::R16_SFLOAT, 1});
    AddTextureToPermanentPool({Format::R8_UNORM, 1});
    AddTextureToPermanentPool({Format::RGBA8_UNORM, 1});
    AddTextureToPermanentPool({Format::R8_UNORM, 1});
    AddTextureToPermanentPool({Format::R32_SFLOAT, 1});

    enum class Transient {
        SPEC_ILLUM_PING = TRANSIENT_POOL_START,
        SPEC_ILLUM_PONG,
        DIFF_ILLUM_PING,
        DIFF_ILLUM_PONG,
        SPEC_REPROJECTION_CONFIDENCE,
        TILES,
        HISTORY_LENGTH
    };

    AddTextureToTransientPool({Format::RGBA16_SFLOAT, 1});
    AddTextureToTransientPool({Format::RGBA16_SFLOAT, 1});
    AddTextureToTransientPool({Format::RGBA16_SFLOAT, 1});
    AddTextureToTransientPool({Format::RGBA16_SFLOAT, 1});
    AddTextureToTransientPool({Format::R8_UNORM, 1});
    AddTextureToTransientPool({Format::R8_UNORM, 16});
    AddTextureToTransientPool({Format::R8_UNORM, 1});

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
        AddDispatch(RELAX_ClassifyTiles, defines);
    }

    for (int i = 0; i < RELAX_HITDIST_RECONSTRUCTION_PERMUTATION_NUM; i++) {
        bool is5x5 = (((i >> 0) & 0x1) != 0);

        PushPass("Hit distance reconstruction");
        {
            // Inputs
            PushInput(AsUint(Transient::TILES));
            PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
            PushInput(AsUint(ResourceType::IN_VIEWZ));
            PushInput(AsUint(ResourceType::IN_SPEC_RADIANCE_HITDIST));
            PushInput(AsUint(ResourceType::IN_DIFF_RADIANCE_HITDIST));

            // Outputs
            PushOutput(AsUint(Transient::SPEC_ILLUM_PING));
            PushOutput(AsUint(Transient::DIFF_ILLUM_PING));

            // Shaders
            std::array<ShaderMake::ShaderConstant, 3> defines = {{
                commonDefines[0],
                commonDefines[1],
                {"MODE_5X5", is5x5 ? "1" : "0"},
            }};
            AddDispatch(RELAX_HitDistReconstruction, defines);
        }
    }

    for (int i = 0; i < RELAX_PREPASS_PERMUTATION_NUM; i++) {
        bool isAfterReconstruction = (((i >> 0) & 0x1) != 0);

        PushPass("Pre-pass");
        {
            // Inputs
            PushInput(AsUint(Transient::TILES));
            PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
            PushInput(AsUint(ResourceType::IN_VIEWZ));
            PushInput(isAfterReconstruction ? AsUint(Transient::SPEC_ILLUM_PING) : AsUint(ResourceType::IN_SPEC_RADIANCE_HITDIST));
            PushInput(isAfterReconstruction ? AsUint(Transient::DIFF_ILLUM_PING) : AsUint(ResourceType::IN_DIFF_RADIANCE_HITDIST));

            // Outputs
            PushOutput(AsUint(ResourceType::OUT_SPEC_RADIANCE_HITDIST));
            PushOutput(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));

            // Shaders
            AddDispatch(RELAX_PrePass, commonDefines);
        }
    }

    for (int i = 0; i < RELAX_TEMPORAL_ACCUMULATION_PERMUTATION_NUM; i++) {
        bool hasDisocclusionThresholdMix = (((i >> 1) & 0x1) != 0);
        bool hasConfidenceInputs = (((i >> 0) & 0x1) != 0);

        PushPass("Temporal accumulation");
        {
            // Inputs
            PushInput(AsUint(Transient::TILES));
            PushInput(AsUint(ResourceType::IN_MV));
            PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
            PushInput(AsUint(ResourceType::IN_VIEWZ));
            PushInput(hasDisocclusionThresholdMix ? AsUint(ResourceType::IN_DISOCCLUSION_THRESHOLD_MIX) : RELAX_DUMMY);
            PushInput(AsUint(Permanent::NORMAL_ROUGHNESS_PREV));
            PushInput(AsUint(Permanent::VIEWZ_PREV));
            PushInput(AsUint(Permanent::HISTORY_LENGTH_PREV));
            PushInput(AsUint(Permanent::MATERIAL_ID_PREV));
            PushInput(AsUint(ResourceType::OUT_SPEC_RADIANCE_HITDIST));
            PushInput(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));
            PushInput(AsUint(Permanent::SPEC_ILLUM_RESPONSIVE_PREV));
            PushInput(AsUint(Permanent::DIFF_ILLUM_RESPONSIVE_PREV));
            PushInput(AsUint(Permanent::SPEC_ILLUM_PREV));
            PushInput(AsUint(Permanent::DIFF_ILLUM_PREV));
            PushInput(AsUint(Permanent::REFLECTION_HIT_T_PREV), AsUint(Permanent::REFLECTION_HIT_T_CURR));
            PushInput(hasConfidenceInputs ? AsUint(ResourceType::IN_SPEC_CONFIDENCE) : RELAX_DUMMY);
            PushInput(hasConfidenceInputs ? AsUint(ResourceType::IN_DIFF_CONFIDENCE) : RELAX_DUMMY);

            // Outputs
            PushOutput(AsUint(Transient::HISTORY_LENGTH));
            PushOutput(AsUint(Transient::SPEC_ILLUM_PING));
            PushOutput(AsUint(Transient::DIFF_ILLUM_PING));
            PushOutput(AsUint(Transient::SPEC_ILLUM_PONG));
            PushOutput(AsUint(Transient::DIFF_ILLUM_PONG));
            PushOutput(AsUint(Permanent::REFLECTION_HIT_T_CURR), AsUint(Permanent::REFLECTION_HIT_T_PREV));
            PushOutput(AsUint(Transient::SPEC_REPROJECTION_CONFIDENCE));

            // Shaders
            AddDispatch(RELAX_TemporalAccumulation, commonDefines);
        }
    }

    PushPass("History fix");
    {
        // Inputs
        PushInput(AsUint(Transient::TILES));
        PushInput(AsUint(Transient::HISTORY_LENGTH));
        PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
        PushInput(AsUint(ResourceType::IN_VIEWZ));
        PushInput(AsUint(Transient::SPEC_ILLUM_PING)); // Normal history
        PushInput(AsUint(Transient::DIFF_ILLUM_PING));

        // Outputs
        PushOutput(AsUint(Transient::SPEC_ILLUM_PONG)); // Responsive history
        PushOutput(AsUint(Transient::DIFF_ILLUM_PONG));

        // Shaders
        AddDispatch(RELAX_HistoryFix, commonDefines);
    }

    PushPass("History clamping");
    {
        // Inputs
        PushInput(AsUint(Transient::TILES));
        PushInput(AsUint(ResourceType::IN_VIEWZ));
        PushInput(AsUint(Transient::HISTORY_LENGTH));
        PushInput(AsUint(ResourceType::OUT_SPEC_RADIANCE_HITDIST)); // Noisy input with preblur applied
        PushInput(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));
        PushInput(AsUint(Transient::SPEC_ILLUM_PING)); // Normal history
        PushInput(AsUint(Transient::DIFF_ILLUM_PING));
        PushInput(AsUint(Transient::SPEC_ILLUM_PONG)); // Responsive history
        PushInput(AsUint(Transient::DIFF_ILLUM_PONG));

        // Outputs
        PushOutput(AsUint(Permanent::HISTORY_LENGTH_PREV));
        PushOutput(AsUint(Permanent::SPEC_ILLUM_PREV));
        PushOutput(AsUint(Permanent::DIFF_ILLUM_PREV));
        PushOutput(AsUint(Permanent::SPEC_ILLUM_RESPONSIVE_PREV));
        PushOutput(AsUint(Permanent::DIFF_ILLUM_RESPONSIVE_PREV));

        // Shaders
        AddDispatch(RELAX_HistoryClamping, commonDefines);
    }

    PushPass("Copy");
    {
        // Inputs
        PushInput(AsUint(Permanent::SPEC_ILLUM_PREV));
        PushInput(AsUint(Permanent::DIFF_ILLUM_PREV));

        // Outputs
        PushOutput(AsUint(ResourceType::OUT_SPEC_RADIANCE_HITDIST));
        PushOutput(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));

        // Shaders
        AddDispatch(RELAX_Copy, commonDefines);
    }

    PushPass("Anti-firefly");
    {
        // Inputs
        PushInput(AsUint(Transient::TILES));
        PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
        PushInput(AsUint(ResourceType::IN_VIEWZ));
        PushInput(AsUint(ResourceType::OUT_SPEC_RADIANCE_HITDIST));
        PushInput(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));

        // Outputs
        PushOutput(AsUint(Permanent::SPEC_ILLUM_PREV));
        PushOutput(AsUint(Permanent::DIFF_ILLUM_PREV));

        // Shaders
        AddDispatch(RELAX_AntiFirefly, commonDefines);
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
                PushInput(AsUint(Transient::HISTORY_LENGTH));
                PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
                PushInput(AsUint(ResourceType::IN_VIEWZ));

                if (isSmem) {
                    PushInput(AsUint(Permanent::SPEC_ILLUM_PREV));
                    PushInput(AsUint(Permanent::DIFF_ILLUM_PREV));
                } else {
                    PushInput(isEven ? AsUint(Transient::SPEC_ILLUM_PONG) : AsUint(Transient::SPEC_ILLUM_PING));
                    PushInput(isEven ? AsUint(Transient::DIFF_ILLUM_PONG) : AsUint(Transient::DIFF_ILLUM_PING));
                }

                PushInput(AsUint(Transient::SPEC_REPROJECTION_CONFIDENCE));
                PushInput(hasConfidenceInputs ? AsUint(ResourceType::IN_SPEC_CONFIDENCE) : RELAX_DUMMY);
                PushInput(hasConfidenceInputs ? AsUint(ResourceType::IN_DIFF_CONFIDENCE) : RELAX_DUMMY);

                // Outputs
                if (isLast) {
                    PushOutput(AsUint(ResourceType::OUT_SPEC_RADIANCE_HITDIST));
                    PushOutput(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));
                } else {
                    PushOutput(isEven ? AsUint(Transient::SPEC_ILLUM_PING) : AsUint(Transient::SPEC_ILLUM_PONG));
                    PushOutput(isEven ? AsUint(Transient::DIFF_ILLUM_PING) : AsUint(Transient::DIFF_ILLUM_PONG));
                }

                if (isSmem) {
                    PushOutput(AsUint(Permanent::NORMAL_ROUGHNESS_PREV));
                    PushOutput(AsUint(Permanent::MATERIAL_ID_PREV));
                    PushOutput(AsUint(Permanent::VIEWZ_PREV));
                }

                // Shaders
                constexpr uint32_t maxRepeatNum = (RELAX_MAX_ATROUS_PASS_NUM - 2 + 1) / 2;
                if (isSmem)
                    AddDispatch(RELAX_AtrousSmem, commonDefines);
                else
                    AddDispatchWithArgs(RELAX_Atrous, commonDefines, 1, maxRepeatNum);
            }
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
        AddDispatch(RELAX_SplitScreen, commonDefines);
    }

    RELAX_ADD_VALIDATION_DISPATCH;
}

#undef DENOISER_NAME
