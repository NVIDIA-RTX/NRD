# HOW TO UPDATE?

This guide has been designed to simplify migration to a newer version.

## To v3.0
*(based on v2.12)*
- `NRD_USE_MATERIAL_ID_AWARE_FILTERING` renamed to `NRD_USE_MATERIAL_ID`
- `NRD_NORMAL_ENCODING` simplified to `NRD_NORMAL_ENCODING_UNORM` and `NRD_NORMAL_ENCODING_OCT`
- `NRD_MATERIAL_ID_AWARE_FILTERING` renamed to `NRD_USE_MATERIAL_ID`
- changed parameters in `NRD_GetCorrectedHitDist`
- *REFERENCE*:
  - denoiser input & output changed to dedicated `IN_RADIANCE` and `OUT_RADIANCE`
- *REBLUR*:
  - settings collapsed to `ReblurSettings` shared across all *REBLUR* denoisers
  - `LobeTrimmingParameters` renamed to `SpecularLobeTrimmingParameters`
  - `normalWeightStrictness` replaced by `lobeAngleFraction` (a different value with similar meaning)
  - `materialMask` replaced with `enableMaterialTestForDiffuse` and `enableMaterialTestForSpecular`
  - exposed `minConvergedStateBaseRadiusScale`
  - exposed `roughnessFraction`
  - exposed `responsiveAccumulationRoughnessThreshold`
  - exposed `inputMix`
  - exposed `enablePerformanceMode`
- *RELAX*:
  - `rejectDiffuseHistoryNormalThreshold` renamed to `diffuseHistoryRejectionNormalThreshold`
  - `enableSkipReprojectionTestWithoutMotion` renamed to `enableReprojectionTestSkippingWithoutMotion`
  - `phiNormal` replaced by `diffuseLobeAngleFraction` (a different value with similar meaning)
  - exposed `roughnessFraction`
  - exposed `enableMaterialTest`

## To v3.1

*NRD* requires explicit definitions of `NRD_USE_OCT_NORMAL_ENCODING` and `NRD_USE_MATERIAL_ID` to avoid a potential confusion when NRD was compiled using one set of values, but an application uses other values. The *CMake* file has been modified to simplify *NRD* integration into projects, where it is used as a *Git* submodule. Now these macro definitions are exposed as *Cmake* options.

## To v3.2

- removed `PrePassMode`
- added `maxSupportedMaterialBitNum` and `isCompiledWithOctPackNormalEncoding` to `LibraryDesc`
- added optional `worldPrevToWorldMatrix` to `CommonSettings`
- *REBLUR*:
  - exposed `diffusePrepassBlurRadius` and `specularPrepassBlurRadius`
  - exposed `enableAdvancedPrepass` matching previously used `PrePassMode::ADVANCED`
  - exposed `enableHitDistanceReconstruction`
- *RELAX*:
  - removed `diffuseHistoryRejectionNormalThreshold`
  - removed `minLuminanceWeight` and added separated `diffuseMinLuminanceWeight` and `specularMinLuminanceWeight`
  - exposed `enableSpecularHitDistanceReconstruction`

## To v3.3

- exposed `HitDistanceReconstructionMode`
- *REBLUR*:
  - `enableHitDistanceReconstruction` replaced with `hitDistanceReconstructionMode`
  - removed `residualNoiseLevel`
- *RELAX*:
  - `enableSpecularHitDistanceReconstruction` replaced with `hitDistanceReconstructionMode`

## To v3.4

- exposed `REBLUR_DIFFUSE_SH`, `REBLUR_SPECULAR_SH` and `REBLUR_DIFFUSE_SPECULAR_SH` denoisers
- *NRD.hlsli*:
  - `REBLUR_FrontEnd_PackRadianceAndHitDist` renamed to `REBLUR_FrontEnd_PackRadianceAndNormHitDist`
  - `REBLUR_FrontEnd_PackDirectionAndHitDist` renamed to `REBLUR_FrontEnd_PackDirectionAndNormHitDist`
  - `REBLUR_BackEnd_UnpackRadianceAndHitDist` renamed to `REBLUR_BackEnd_UnpackRadianceAndNormHitDist`
  - `REBLUR_BackEnd_UnpackDirectionAndHitDist` renamed to `REBLUR_BackEnd_UnpackDirectionAndNormHitDist`
  - `RELAX_BackEnd_UnpackRadianceAndHitDist` renamed to `RELAX_BackEnd_UnpackRadiance`
  - removed `NRD_GetCorrectedHitDist`
- *API*:
  - exposed miscellaneous function `GetMethodString`
- *REBLUR*:
  - removed `inputMix`
- *RELAX*:
  - exposed `confidenceDrivenRelaxationMultiplier`
  - exposed `confidenceDrivenLuminanceEdgeStoppingRelaxation`
  - exposed `confidenceDrivenNormalEdgeStoppingRelaxation`

## To v3.5

- *NRD.hlsli*:
  - `REBLUR_FrontEnd_PackDirectionAndNormHitDist` renamed to `REBLUR_FrontEnd_PackDirectionalOcclusion`
- *REBLUR*:
  - exposed `historyFixFrameNum`
  - exposed `historyFixStrideBetweenSamples`
  - removed `historyFixStrength`
- *RELAX*:
  - `disocclusionFixEdgeStoppingNormalPower` renamed to `historyFixEdgeStoppingNormalPower`
  - `disocclusionFixMaxRadius` renamed to `historyFixStrideBetweenSamples`
  - `disocclusionFixNumFramesToFix` renamed to `historyFixFrameNum`

## To v3.6

- *REBLUR*:
  - Removed `SpecularLobeTrimmingParameters`
- *NRD.hlsli*:
  - Removed `NRD_GetTrimmingFactor`

## To v3.7

`NRD_USE_OCT_NORMAL_ENCODING` and `NRD_USE_MATERIAL_ID` replaced with more explicit `NRD_NORMAL_ENCODING` and `NRD_ROUGHNESS_ENCODING` to offer more encoding variants out of the box.

- *API*:
  - removed `maxSupportedMaterialBitNum` and `isCompiledWithOctPackNormalEncoding` from `LibraryDesc`
  - added `normalEncoding` and `roughnessEncoding` to `LibraryDesc`
- *REBLUR*:
  - Removed `enableAdvancedPrepass`

## To v3.8

Introduced optional `NRDEncoding.hlsli` file which can be included prior `NRD.hlsli` to properly setup NRD encoding for usage outside of NRD project, if encoding variants are not set via Cmake parameters. Also introduced optional `IN_DISOCCLUSION_THRESHOLD_MIX` which allows to smoothly mix `CommonSettings::disocclusionThreshold` (0) into `CommonSettings::disocclusionThresholdAlternate` (1).

- *API*:
  - Exposed `IN_DISOCCLUSION_THRESHOLD_MIX`, `CommonSettings::disocclusionThresholdAlternate` & `CommonSettings::isDisocclusionThresholdMixAvailable`
- *REBLUR*:
  - Removed `minConvergedStateBaseRadiusScale`
  - Removed `maxAdaptiveRadiusScale`

## To v3.9

Introduced optional `OUT_VALIDATION` output, which contains debug visualization layer if `CommonSettings::enableValidation = true`.

- *API*:
  - Removed `DenoiserCreationDesc::enableValidation`
  - Exposed `CommonSettings::enableValidation`
  - Extened `CommonSettings::motionVectorScale` to 3 floats

## To v4.0

Since *NRD* tracks specular motion, now, if requested, it can modify provided *diffuse*-like motion in `IN_MV` with internally computed *specular*-like motion if specularity is high. For this purpose an optional `IN_BASECOLOR_METALNESS` input has been added. This feature improves behavior of spatio-temporal upscalers, like *TAA* or *DLSS*.

- *API*:
  - Introduced `CommonSettings::isBaseColorMetalnessAvailable`
  - Reworked `DenoiserDesc` to clearly indicate that there are 3 types of resources each of which "sits" in a predefined `space` (`set` in *VK*):
    - constant buffer - binding is shared across all pipelines
    - samplers - bindings are shared across all pipelines
    - resources - bindings vary per pipeline
  - Name changes:
    - `isHistoryConfidenceInputsAvailable` => `isHistoryConfidenceAvailable`
    - `Resource` => `ResourceDesc`
    - `DescriptorRangeDesc` => `ResourceRangeDesc`
    - `ComputeShader` => `ComputeShaderDesc`
    - `DescriptorSetDesc` => `DescriptorPoolDesc`
    - all `fooNum` => `foosNum`
  - `GetComputeDispatches` return type changed to `void`
- *REBLUR*:
  - Exposed `specularProbabilityThresholdsForMvModification` to control diffuse / specular motion mixing
- *RELAX*:
  - Removed `enableSpecularVirtualHistoryClamping`

## To v4.1

SH (spherical harmonics) have been replaced with SG (spherical gaussians). It unlocks physically-based high quality diffuse & specular resolve. The following functions have been added to `NRD.hlsli` (or renamed):
- `NRD_SG_ExtractColor` - extracts unresolved denoised radiance
- `NRD_SG_ExtractDirection` - extracts light dominant direction
- `NRD_SG_ExtractRoughnessAA` - extracts modified roughness (increased in areas with high normal variance)
- `NRD_SG_ResolveDiffuse` - reconstructs diffuse macro details
- `NRD_SG_ResolveSpecular` - reconstructs specular macro details
- `NRD_SG_ReJitter` - reconstructs diffuse & specular micro details (resurrects jitter vanished out after denoising)
- `NRD_SH_ResolveDiffuse` - reconstructs diffuse macro details using SH resolve (for comparison only)
- `NRD_SH_ResolveSpecular` - reconstructs specular macro details using SH resolve (for comparison only)

## To v4.2

A single NRD instance can now include any combination of denoisers, including repeating ones (for example, `RELAX_DIFFUSE`, `RELAX_DIFFUSE` and `SIGMA_SHADOW`). Transient memory pool is internally optimized to reduce memory consumption.

- *API*:
  - Introduced `Identifier` to distinguish one denoiser in the instance from another one
  - Introduced `SetCommonSettings`
  - `GetComputeDispatches` now expects a list of identifiers, specifying which denoisers collect dispatches for
  - Added explicit `CommonSettings::cameraJitterPrev` and `CommonSettings::resolutionScalePrev` (since `SetCommonSettings` can be called several times per frame)
  - Name changes (including sub-name usages):
    - `Denoiser` => `Instance`
    - `Method` => `Denoiser`
    - `fullResolutionWidth` => `renderWidth`
    - `fullResolutionHeight` => `renderHeight`
- *NRD INTEGRATION*:
  - Introduced `NewFrame`
  - Introduced `SetCommonSettings`

## To v4.3

- *RELAX*:
  - Introduced `AntilagSettings`
- *REBLUR*:
  - Removed `enableReferenceAccumulation`
  - Introduced `usePrepassOnlyForSpecularMotionEstimation`
  - `AntilagIntensitySettings` and `AntilagHitDistanceSettings` replaced with simpler `ReblurAntilagSettings`

## To v4.4

- *API*:
  - `inputSubrectOrigin` => `rectOrigin` and clarified usage
  - improved dynamic resolution scaling support: now not only viewport can be adjusted on the fly, but resources can be suballocated from a heap on the fly too
    - removed `DenoiserDesc::renderWidth` and `DenoiserDesc::renderHeight`
    - removed `CommonSettings::resolutionScale` and `CommonSettings::resolutionScalePrev`
    - exposed `CommonSettings::resourceSize` and `CommonSettings::resourceSizePrev`
    - exposed `CommonSettings::rectSize` and `CommonSettings::rectSizePrev`
    - *NRD INTEGRATION*: currently statically allocates resources at NRD instance creation time
  - all denoisers use resources without mips:
    - removed `TextureDesc::mipNum`
    - removed `ResourceDesc::mipNum` and `ResourceDesc::mipOffset`
  - simplified number of samplers
    - removed `Sampler::NEAREST_MIRRORED_REPEAT` and `Sampler::LINEAR_MIRRORED_REPEAT`
- *NRD.hlsli*:
  - added `NRD_FrontEnd_SpecHitDistAveraging_X` functions, needed to properly average specular hit distance for correct specular tracking in case of many RPP
  - `NRD_INPUT/OUTPUT_TEXTURE_START` renamed to `NRD_INPUTS/OUTPUTS_START`
  - `NRD_INPUT/OUTPUT_TEXTURE` renamed to `NRD_INPUT/OUTPUT`
  - `NRD_INPUT/OUTPUT_TEXTURE_END` renamed to `NRD_INPUTS/OUTPUTS_END`
  - `NRD_SAMPLER_START/END` renamed to `NRD_SAMPLERS_START/END`
- *REBLUR*:
  - removed `historyFixStrideBetweenSamples`
- *RELAX*:
  - removed `historyFixStrideBetweenSamples`
  - removed `enableReprojectionTestSkippingWithoutMotion`
- *REFERENCE*:
  - `IN_RADIANCE` renamed to `IN_SIGNAL`
  - `OUT_RADIANCE` renamed to `OUT_SIGNAL`

## To v4.5

- *API*:
  - `ResourceDesc::stateNeeded` renamed to `ResourceDesc::descriptorType` to match `ResourceRangeDesc::descriptorType` because the former is a concatenation of the latter

## To v4.6

- *REBLUR*:
  - `blurRadius` renamed to `maxBlurRadius`
  - exposed `minBlurRadius` with the default value matching older versions

## To v4.7

- *SIGMA*:
  - removed `blurRadiusScale`
  - exposed `lightDirection`, which is needed only for directional light sources
  - exposed `stabilizationStrength'
  - clarified usage:
    - `float shadow = SIGMA_BackEnd_UnpackShadow( OUT_SHADOW_TRANSLUCENCY );`
    - `float3 translucentShadow = SIGMA_BackEnd_UnpackShadow( OUT_SHADOW_TRANSLUCENCY ).yzw;`

## To v4.8

- *API*:
  - exposed `CommonSettings::viewZScale` improving FP16 `viewZ` support
- *SIGMA*:
  - explicitly uses `IN_VIEWZ` instead of storing `viewZ` in `IN_SHADOWDATA`
  - `IN_SHADOWDATA` renamed to `IN_PENUMBRA` (describes penumbra properties)
  - `IN_SHADOW_TRANSLUCENCY` renamed to `IN_TRANSLUCENCY` (describes optional translucency)
  - `SIGMA_FrontEnd_PackShadow` has been explicitly separated into 2 functions:
    - `IN_SHADOWDATA`: `SIGMA_FrontEnd_PackPenumbra` (2 variants: for infinite and local light sources)
    - `IN_SHADOW_TRANSLUCENCY`: `SIGMA_FrontEnd_PackTranslucency`
- Deprecated and removed with all dependencies:
  - `SPECULAR_REFLECTION_MV`
  - `SPECULAR_DELTA_MV`

## To v4.9

- *API*:
  - `MemoryAllocatorInterface` renamed to `AllocationCallbacks`
  - exposed `CommonSettings::strandMaterialID`, enabling "under-the-hood" tweaks for hair (and grass) denoising
  - exposed `CommonSettings::strandThickness`, defining how NRD adapts to sub-pixel thick details. It works in conjunction with `CommonSettings::disocclusionThresholdAlternate` for `CommonSettings::strandMaterialID` without a need to provide `IN_DISOCCLUSION_THRESHOLD_MIX` texture
- *REBLUR*:
  - changed usage of `maxBlurRadius` and its default value, old values should be scaled by `2`
  - exposed `hitDistanceStabilizationStrength` allowing to control the responsiviness of ambient and specular occlusion in the temporal stabilization pass and reach parity with `REBLUR_OCCLUSION` if set to `0`

## To v4.10

- *NRD INTEGRATION*:
  - `Nrd` prefix replaced with `nrd::` namespace
  - removed `format` from `NrdIntegrationTexture`, which is now just `nri::TextureBarrierDesc`
  - removed the constructor with arguments
  - creation parameters grouped into `IntegrationCreationDesc`, which need to be passed to `Initialize`
  - added a few potentially useful flags to `IntegrationCreationDesc`

## To v4.11
- *API*:
  - exposed `DEFAULT_ACCUMULATION_TIME` for all denoisers
  - added some missing `MAX_HISTORY_FRAME_NUM` constants
  - exposed `GetMaxAccumulatedFrameNum` helper, converting time to frames
- *REBLUR*:
  - `stabilizationStrength` replaced with `maxStabilizedFrameNum`
  - `hitDistanceStabilizationStrength` replaced with `maxStabilizedFrameNumForHitDistance`
  - `luminanceAntilagPower` replaced with `luminanceSensitivity`
  - `hitDistanceAntilagPower` replaced with `hitDistanceSensitivity`
  - tweaked the default values for `ReblurAntilagSettings`
  - the new default value for `planeDistanceSensitivity = 0.02` matches the behavior for the old value `0.005`
  - slightly reduced `lobeAngleFraction` to squeeze more normal details (it's safe to use the old value `0.2`)
- *SIGMA*:
  - `stabilizationStrength` replaced with `maxStabilizedFrameNum`
  - the new default value for `planeDistanceSensitivity = 0.02` matches the behavior for the old value `0.005`

## To v4.12
- *REBLUR/RELAX*:
  - (re)introduced `historyFixBasePixelStride` to maximize flexibility (the default matches old behavior)

## To v4.13
- *REBLUR/RELAX*:
  - added `minHitDistanceWeight`

## To v4.14
- *REBLUR/RELAX*:
  - `enableMaterialTestForDiffuse` replaced with `minMaterialForDiffuse` (the default matches old behavior)
  - `enableMaterialTestForSpecular` replaced with `minMaterialForSpecular` (the default matches old behavior)
- *REBLUR*:
  - Disabled temporal stabilization for `hitT`:
    - removed `ReblurAntilagSettings::hitDistanceSigmaScale` and `ReblurAntilagSettings::hitDistanceSensitivity`
    - removed `maxStabilizedFrameNumForHitDistance`
  - output textures are not used as history buffers on the next frame anymore

## To v4.15
- *API*:
  - removed `ResourceRangeDesc::baseRegisterIndex`and added global `InstanceDesc::resourcesBaseRegisterIndex`
  - spaces for constant buffer and resources merged into a single `InstanceDesc::constantBufferAndResourcesSpaceIndex`
  - reworked `DescriptorPoolDesc`

## To v4.16
- *API*:
  - `NRDEncoding.hlsli` renamed to `NRDConfig.hlsli` and included into `NRD.hlsli` (now includes more compile time constants)
  - removed `DescriptorPoolDesc::totalConstantBuffersNum` and `DescriptorPoolDesc::totalSamplersNum`. If root (immutable) samplers and root (push) descriptor are not used, this can be deduced from `DescriptorPoolDesc::setMaxNum` and clarifying comments
  - merged register spaces for samplers and constant buffers into `InstanceDesc::constantBufferAndSamplersSpaceIndex`
  - renamed register space for resources (SRVs and UAVs) to `InstanceDesc::resourcesSpaceIndex`
  - separated base registers for constant buffer, samplers and resources in `InstanceDesc`
  - removed `InstanceDesc::samplersInSeparateSet`
  - moved `PipelineDesc::shaderEntryPoint` to `InstanceDesc`
  - added `CommonSettings::historyFixAlternatePixelStrideMaterialID` allowing to use `historyFixAlternatePixelStride` for a specific material ID
  - replaced `PipelineDesc::shaderFileName` with `shaderIdentifier` (including shader name and permutation macros)
  - reworked `GetLibraryDesc` and `GetInstanceDesc` to return a pointer instead of a reference to be conformant with "C" linkage
- *NRD INTEGRATION*:
  - expects unique register spaces
- *REBLUR*:
  - added `returnHistoryLengthInsteadOfOcclusion`
  - added `historyFixAlternatePixelStride`
  - exposed previously hidden `fastHistoryClampingSigmaScale` (the default matches old behavior)
  - responsive accumulation settings extended and grouped into `ResponsiveAccumulationSettings`
- *RELAX*:
  - diffuse history length is returned in ".w"
  - added `historyFixAlternatePixelStride`
  - `historyClampingColorBoxSigmaScale` renamed to `fastHistoryClampingSigmaScale` (no changes in meaning)

## Custom shader compilation
- since v4.16 `NRDConfig.hlsli` is included into every shader (including `NRD.hlsli`), delivering shared compile-time options. So there is no need to "copy-paste" anything from CMake
- `PipelineDesc::shaderIdentifier` defines a shader permutation as `fileName|macro1=value1|macro2=value2...` (or just `fileName`, no whitespace characters)
- the entry point is provided in `InstanceDesc::shaderEntryPoint` (same as `NRD_CS_MAIN` macro)
- an array of macro definitions needs to be passed to a custom shader build system alongside with the file name
- (offline) an application can loop through all pipelines in the NRD instance to gather all permutations and compile them via a custom shader build system
- (runtime) `PipelineDesc::shaderIdentifier` should be used to identify and provide a shader permutation NRD is asking for, since embedded shaders can't be used
