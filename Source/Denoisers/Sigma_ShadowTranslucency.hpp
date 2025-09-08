/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#define DENOISER_NAME SIGMA_ShadowTranslucency

void nrd::InstanceImpl::Add_SigmaShadowTranslucency(nrd::DenoiserData& denoiserData) {
    denoiserData.settings.sigma = SigmaSettings();
    denoiserData.settingsSize = sizeof(denoiserData.settings.sigma);

    enum class Permanent {
        HISTORY_LENGTH = PERMANENT_POOL_START,
    };

    AddTextureToPermanentPool({Format::R32_UINT, 1});

    enum class Transient {
        DATA_1 = TRANSIENT_POOL_START,
        DATA_2,
        TEMP_1,
        TEMP_2,
        HISTORY,
        HISTORY_LENGTH,
        TILES,
        SMOOTHED_TILES,
    };

    AddTextureToTransientPool({Format::R16_SFLOAT, 1});
    AddTextureToTransientPool({Format::R16_SFLOAT, 1});
    AddTextureToTransientPool({Format::RGBA8_UNORM, 1});
    AddTextureToTransientPool({Format::RGBA8_UNORM, 1});
    AddTextureToTransientPool({Format::RGBA8_UNORM, 1});
    AddTextureToTransientPool({Format::R32_UINT, 1});
    AddTextureToTransientPool({Format::RGBA8_UNORM, 16});
    AddTextureToTransientPool({Format::RG8_UNORM, 16});

    std::array<ShaderMake::ShaderConstant, 1> commonDefines = {{
        {"TRANSLUCENCY", "1"},
    }};

    PushPass("Classify tiles");
    {
        // Inputs
        PushInput(AsUint(ResourceType::IN_VIEWZ));
        PushInput(AsUint(ResourceType::IN_PENUMBRA));
        PushInput(AsUint(ResourceType::IN_TRANSLUCENCY));

        // Outputs
        PushOutput(AsUint(Transient::TILES));

        // Shaders
        AddDispatch(SIGMA_ClassifyTiles, commonDefines);
    }

    PushPass("Smooth tiles");
    {
        // Inputs
        PushInput(AsUint(Transient::TILES));

        // Outputs
        PushOutput(AsUint(Transient::SMOOTHED_TILES));

        // Shaders
        std::array<ShaderMake::ShaderConstant, 0> defines = {};
        AddDispatchWithArgs(SIGMA_SmoothTiles, defines, 16, 1);
    }

    PushPass("Copy");
    {
        // Inputs
        PushInput(AsUint(Transient::SMOOTHED_TILES));
        PushInput(AsUint(ResourceType::OUT_SHADOW_TRANSLUCENCY));
        PushInput(AsUint(Permanent::HISTORY_LENGTH));

        // Outputs
        PushOutput(AsUint(Transient::HISTORY));
        PushOutput(AsUint(Transient::HISTORY_LENGTH));

        // Shaders
        std::array<ShaderMake::ShaderConstant, 0> defines = {};
        AddDispatchWithArgs(SIGMA_Copy, defines, USE_PREV_DIMS, 1);
    }

    PushPass("Blur");
    {
        // Inputs
        PushInput(AsUint(ResourceType::IN_VIEWZ));
        PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
        PushInput(AsUint(ResourceType::IN_PENUMBRA));
        PushInput(AsUint(Transient::SMOOTHED_TILES));
        PushInput(AsUint(ResourceType::IN_TRANSLUCENCY));

        // Outputs
        PushOutput(AsUint(Transient::DATA_1));
        PushOutput(AsUint(Transient::TEMP_1));

        // Shaders
        std::array<ShaderMake::ShaderConstant, 2> defines = {{
            commonDefines[0],
            {"FIRST_PASS", "1"},
        }};
        AddDispatch(SIGMA_Blur, defines);
    }

    for (int i = 0; i < SIGMA_POST_BLUR_PERMUTATION_NUM; i++) {
        bool isStabilizationEnabled = (((i >> 0) & 0x1) != 0);

        PushPass("Post-blur");
        {
            // Inputs
            PushInput(AsUint(ResourceType::IN_VIEWZ));
            PushInput(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
            PushInput(AsUint(Transient::DATA_1));
            PushInput(AsUint(Transient::SMOOTHED_TILES));
            PushInput(AsUint(Transient::TEMP_1));

            // Outputs
            PushOutput(AsUint(Transient::DATA_2));
            PushOutput(isStabilizationEnabled ? AsUint(Transient::TEMP_2) : AsUint(ResourceType::OUT_SHADOW_TRANSLUCENCY));

            // Shaders
            std::array<ShaderMake::ShaderConstant, 2> defines = {{
                commonDefines[0],
                {"FIRST_PASS", "0"},
            }};
            AddDispatch(SIGMA_Blur, defines);
        }
    }

    PushPass("Temporal stabilization");
    {
        // Inputs
        PushInput(AsUint(ResourceType::IN_VIEWZ));
        PushInput(AsUint(ResourceType::IN_MV));
        PushInput(AsUint(Transient::DATA_2));
        PushInput(AsUint(Transient::TEMP_2));
        PushInput(AsUint(Transient::HISTORY));
        PushInput(AsUint(Transient::HISTORY_LENGTH));
        PushInput(AsUint(Transient::SMOOTHED_TILES));

        // Outputs
        PushOutput(AsUint(ResourceType::OUT_SHADOW_TRANSLUCENCY));
        PushOutput(AsUint(Permanent::HISTORY_LENGTH));

        // Shaders
        AddDispatch(SIGMA_TemporalStabilization, commonDefines);
    }

    PushPass("Split screen");
    {
        // Inputs
        PushInput(AsUint(ResourceType::IN_VIEWZ));
        PushInput(AsUint(ResourceType::IN_PENUMBRA));
        PushInput(AsUint(ResourceType::IN_TRANSLUCENCY));

        // Outputs
        PushOutput(AsUint(ResourceType::OUT_SHADOW_TRANSLUCENCY));

        // Shaders
        AddDispatch(SIGMA_SplitScreen, commonDefines);
    }
}

#undef DENOISER_NAME
