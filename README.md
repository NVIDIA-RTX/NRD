# NVIDIA REAL-TIME DENOISERS v4.17.0 (NRD)

[![Build NRD SDK](https://github.com/NVIDIA-RTX/NRD/actions/workflows/build.yml/badge.svg)](https://github.com/NVIDIA-RTX/NRD/actions/workflows/build.yml)

![Title](Images/Title.jpg)

For quick starting see *[NRD sample](https://github.com/NVIDIA-RTX/NRD-Sample)* project:
- `simplex` branch - focuses on path tracing and NRD best practices (less code, less preprocessor, easier to follow)
- `main` branch - contains everything needed for NRD development, testing and maintaining, all variants of NRD usage are here

# OVERVIEW

*NVIDIA Real-Time Denoisers (NRD)* is a spatio-temporal API agnostic denoising library. The library has been designed to work with low rpp (ray per pixel) signals. *NRD* is a fast solution that slightly depends on input signals and environment conditions.

*NRD* includes the following denoisers:
- *REBLUR* - recurrent blur based denoiser
- *RELAX* - A-trous based denoiser, has been designed for *[RTXDI (RTX Direct Illumination)](https://developer.nvidia.com/rtxdi)*
- *SIGMA* - shadow-only denoiser

Performance on RTX 4080 @ 1440p (native resolution, default denoiser settings, `NormalEncoding::R10_G10_B10_A2_UNORM`):
- `REBLUR_DIFFUSE_SPECULAR` - 2.25 ms (3.10 ms in `SH` mode, 2.00 ms in performance mode)
- `RELAX_DIFFUSE_SPECULAR` - 3.00 ms (4.90 ms in `SH` mode)
- `SIGMA_SHADOW` - 0.40 ms
- `SIGMA_SHADOW_TRANSLUCENCY` - 0.45 ms
- memory usage [table](#memory-usage)

Supported signal types:
- *RELAX*:
  - Diffuse & specular radiance
- *REBLUR*:
  - Diffuse & specular radiance
  - Diffuse (ambient) & specular occlusion (OCCLUSION variants)
  - Diffuse (ambient) directional occlusion (DIRECTIONAL_OCCLUSION variant)
  - Diffuse & specular radiance in spherical harmonics (spherical gaussians) (SH variants)
- *SIGMA*:
  - Shadows from an infinite light source (sun, moon)
  - Shadows from a local light source (omni, spot)

For diffuse and specular signals de-modulated irradiance (i.e. irradiance with "removed" materials) can be used instead of radiance (see this [section](#material-demodulation)).

*NRD* is distributed as a source as well with a “ready-to-use” library (if used in a precompiled form). It can be integrated into any DX12, VULKAN or DX11 engine using two variants:
1. Native implementation of the *NRD* API using engine capabilities
2. Integration via an abstraction layer. In this case, the engine should expose native GAPI pointers for certain types of objects. The integration layer, provided as a part of SDK, can be used to simplify this integration variant.

## HOW TO BUILD?

- Install [*Cmake*](https://cmake.org/download/) 3.22+
- Build (variant 1) - using *Git* and *CMake* explicitly
  - Clone project and init submodules
  - Generate and build the project using *CMake*
  - To build the binary with static MSVC runtime, add `-DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>"` parameter when deploying the project
- Build (variant 2) - by running scripts:
  - Run `1-Deploy`
  - Run `2-Build`

CMake options:
- Common:
  - `NRD_NRI` - pull, build and include *NRI* into *NRD SDK* package (OFF by default)
  - `NRD_SHADERS_PATH` - shader output path override
  - `NRD_EMBEDS_DXBC_SHADERS` - *NRD* compiles and embeds DXBC shaders (ON by default on Windows)
  - `NRD_EMBEDS_DXIL_SHADERS` - *NRD* compiles and embeds DXIL shaders (ON by default on Windows)
  - `NRD_EMBEDS_SPIRV_SHADERS` - *NRD* compiles and embeds SPIRV shaders (ON by default)
- Compile time switches (prefer to disable unused functionality to increase performance):
  - `NRD_STATIC_LIBRARY` - build static library (OFF by default, visible in the parent project)
  - `NRD_NORMAL_ENCODING` - *normal* encoding for the entire library
  - `NRD_ROUGHNESS_ENCODING` - *roughness* encoding for the entire library
  - `NRD_SUPPORTS_VIEWPORT_OFFSET` - enable `CommonSettings::rectOrigin` support (OFF by default)
  - `NRD_SUPPORTS_CHECKERBOARD` - enable `checkerboardMode` support (ON by default)
  - `NRD_SUPPORTS_HISTORY_CONFIDENCE` - enable `IN_DIFF_CONFIDENCE` and `IN_SPEC_CONFIDENCE` support (ON by default)
  - `NRD_SUPPORTS_DISOCCLUSION_THRESHOLD_MIX` - enable `IN_DISOCCLUSION_THRESHOLD_MIX` support (ON by default)
  - `NRD_SUPPORTS_BASECOLOR_METALNESS` - enable `IN_BASECOLOR_METALNESS` support (ON by default)
  - `NRD_SUPPORTS_ANTIFIREFLY` - enable `enableAntiFirefly` support (ON by default)
  - `REBLUR_PERFORMANCE_MODE` - better performance and worse image quality, can be useful for consoles (OFF by default)

`NRD_NORMAL_ENCODING` and `NRD_ROUGHNESS_ENCODING` can be defined only *once* during project deployment. `LibraryDesc` includes encoding settings too. It can be used to verify that the library meets the application expectations.

SDK packaging:
- Compile the solution (*Debug* / *Release* or both, depending on what you want to get in *NRD* package)
- Run `3-PrepareSDK`
- Grab generated in the root directory `_NRD_SDK` and `_NRI_SDK` (if needed) folders and use them in your project

## HOW TO UPDATE?

- Clone latest
- Run `4-Clean`
- Run `1-Deploy`
- Run `2-Build`

## HOW TO REPORT ISSUES?

NRD sample has *TESTS* section in the bottom of the UI, a new test can be added if needed. The following procedure is recommended:
- Try to reproduce a problem in the *NRD sample* first
  - if reproducible
    - add a test (by pressing `Add` button)
    - describe the issue and steps to reproduce on *GitHub*
    - attach depending on the selected scene `.bin` file from the `Tests` folder
  - if not
    - verify the integration
- If nothing helps
  - describe the issue, attach a video and steps to reproduce

# API

Terminology:
* *Denoiser* - a denoiser to use (for example: `Denoiser::REBLUR_DIFFUSE`)
* *Instance* - a set of denoisers aggregated into a monolithic entity (the library is free to rearrange passes without dependencies). Each denoiser in the instance has an associated *Identifier*
* *Resource* - an input, output or internal resource (currently can only be a texture)
* *Texture pool (or pool)* - a texture pool that stores permanent or transient resources needed for denoising. Textures from the permanent pool are dedicated to *NRD* and can not be reused by the application (history buffers are stored here). Textures from the transient pool can be reused by the application right after denoising. *NRD* doesn’t allocate anything. *NRD* provides resource descriptions, but resource creations are done on the application side.

Flow:
1. *GetLibraryDesc* - contains general *NRD* library information (supported denoisers, SPIRV binding offsets). This call can be skipped if this information is known in advance (for example, is diffuse denoiser available?), but it can’t be skipped if SPIRV binding offsets are needed for VULKAN
2. *CreateInstance* - creates an instance for requested denoisers
3. *GetInstanceDesc* - returns descriptions for pipelines, samplers, texture pools, constant buffer and descriptor set. All this stuff is needed during the initialization step
4. *SetCommonSettings* - sets common (shared) per frame parameters
5. *SetDenoiserSettings* - can be called to change parameters dynamically before applying the denoiser on each new frame / denoiser call
6. *GetComputeDispatches* - returns per-dispatch data for the list of denoisers (bound subresources with required state, constant buffer data). Returned memory is owned by the instance and gets overwritten by the next *GetComputeDispatches* call
7. *DestroyInstance* - destroys an instance

*NRD* doesn't make any GAPI calls. The application is supposed to invoke a set of compute *Dispatch* calls to do denoising. Refer to `NRDIntegration.hpp` file as an example of an integration using low level RHI.

*NRD* doesn't have a "resize" functionality. On a resolution change the old denoiser needs to be destroyed and a new one needs to be created with new parameters. But *NRD* supports dynamic resolution scaling via `CommonSettings::resourceSize, resourceSizePrev, rectSize, rectSizePrev`.

Some textures can be requested as inputs or outputs for a method. Required resources are specified near a denoiser declaration inside the `Denoiser` enum class. Also `NRD.hlsli` has a comment near each front-end or back-end function, clarifying which resources this function is for.

## INTEGRATION

If GAPI's native pointers are retrievable from the RHI, the [NRDIntegration](https://github.com/NVIDIA-RTX/NRD/blob/master/Integration/NRDIntegration.h) layer can be used to greatly simplify the integration. In this case, the application should only provide native pointers for the *Device*, *CommandList* and *Textures* into entities, compatible with an API abstraction layer (*[NRI](https://github.com/NVIDIA-RTX/NRI)*), and all work with *NRD* library will be hidden inside the integration layer:

*Engine or App → native objects → NRD integration layer → NRI → NRD*

*NRI = NVIDIA Rendering Interface* - an abstraction layer on top of GAPIs: DX11, DX12 and VULKAN. *NRI* has been designed to provide low overhead access to the GAPIs and simplify development of DX12 and VULKAN applications. *NRI* API has been influenced by VULKAN as the common denominator among these 3 APIs.

*NRI* and *NRD* are ready-to-use products. The application must expose native pointers only for Device, Resource and CommandList entities (no SRVs and UAVs - they are not needed, everything will be created internally). Native resource pointers are needed only for the denoiser inputs and outputs (all intermediate textures will be handled internally). Descriptor heap will be changed to an internal one, so the application needs to bind its original descriptor heap after invoking the denoiser.

In rare cases, when the integration via the engine’s RHI is not possible and the integration using native pointers is complicated, a "DoDenoising" call can be added explicitly to the application-side RHI. It helps to avoid increasing code entropy.

<details>
<summary>An example demonstrating how to use "NRDIntegration" layer (CLICK):</summary>

```cpp
//=======================================================================================================
// DECLARATIONS (using D3D12 as an example)
//=======================================================================================================

#include "NRI.h"
#include "Extensions/NRIHelper.h"
#include "Extensions/NRIWrapperD3D12.h" // VK, D3D11 (all of them)

#include "NRD.h"
#include "NRDIntegration.hpp"

nrd::Integration NRD = {};

// Converts an app-side texture into an NRD resource
nrd::Resource GetNrdResource(MyTexture& myTexture) {
    nrd::Resource resource = {};
    resource.d3d12.resource = myTexture.GetD3D12Resource();
    resource.d3d12.format = myTexture.GetFormat();
    resource.userArg = &myTexture;
    resource.state = myTexture->state; // "last after" state

    return resource;
}

//=======================================================================================================
// INITIALIZATION
//=======================================================================================================

nri::QueueFamilyD3D12Desc queueFamilyD3D12Desc = {};
queueFamilyD3D12Desc.d3d12Queues = &d3d12Queue;
queueFamilyD3D12Desc.queueNum = 1;
queueFamilyD3D12Desc.queueType = nri::QueueType::GRAPHICS; // or COMPUTE

nri::DeviceCreationD3D12Desc deviceCreationD3D12Desc = {};
deviceCreationD3D12Desc.d3d12Device = d3d12Device;
deviceCreationD3D12Desc.queueFamilies = &queueFamilyD3D12Desc;
deviceCreationD3D12Desc.queueFamilyNum = 1;

const nrd::DenoiserDesc denoiserDescs[] =
{
    // Put needed denoisers here...
    { identifier1, nrd::Denoiser::AAA },
    { identifier2, nrd::Denoiser::BBB },
};

nrd::InstanceCreationDesc instanceCreationDesc = {};
instanceCreationDesc.denoisers = denoiserDescs;
instanceCreationDesc.denoisersNum = 2;

nrd::IntegrationCreationDesc integrationCreationDesc = {};
strncpy(integrationCreationDesc.name, "NRD", sizeof(integrationCreationDesc.name));
integrationCreationDesc.queuedFrameNum = 3; // i.e. number of frames "in-flight"
integrationCreationDesc.enableWholeLifetimeDescriptorCaching = false; // safer, but unrecommended
integrationCreationDesc.autoWaitForIdle = true; // for lazy people

// NRD itself is flexible and supports any kind of dynamic resolution scaling, but NRD INTEGRATION pre-
// allocates resources with statically defined dimensions. DRS is only supported by adjusting the viewport
// via "CommonSettings::rectSize"
integrationCreationDesc.resourceWidth = resourceWidth;
integrationCreationDesc.resourceHeight = resourceHeight;

// Also NRD needs to be recreated on "resize"
nrd::Result result = NRD.RecreateD3D12(integrationCreationDesc, instanceCreationDesc, deviceCreationD3D12Desc);

//=======================================================================================================
// PREPARE
//=======================================================================================================

// Must be called once on a frame start
NRD.NewFrame();

// Set common settings
nrd::CommonSettings commonSettings = {};
PopulateCommonSettings(commonSettings);

NRD.SetCommonSettings(commonSettings);

// Set settings for denoisers
nrd::AaaSettings settings1 = {};
PopulateAaaSettings(settings1);

NRD.SetDenoiserSettings(identifier1, &settings1);

nrd::BbbSettings settings2 = {};
PopulateBbbSettings(settings2);

NRD.SetDenoiserSettings(identifier2, &settings2);

//=======================================================================================================
// RENDER
//=======================================================================================================

// Fill resource snapshot
nrd::ResourceSnapshot resourceSnapshot = {};
{
    resourceSnapshot.restoreInitialState = true; // simpler, but unrecommended

    // Common
    resourceSnapshot.SetResource(nrd::ResourceType::IN_MV, GetNrdResource(myTexture_Mv));
    resourceSnapshot.SetResource(nrd::ResourceType::IN_NORMAL_ROUGHNESS, GetNrdResource(myTexture_NormalRoughness));
    resourceSnapshot.SetResource(nrd::ResourceType::IN_VIEWZ, GetNrdResource(myTexture_ViewZ));

    // Denoiser specific
    ...
}

// Denoise
nri::CommandBufferD3D12Desc commandBufferD3D12Desc = {};
commandBufferD3D12Desc.d3d12CommandList = d3d12CommandList;

const nrd::Identifier denoisers[] = {identifier1, identifier2};
m_NRD.DenoiseD3D12(denoisers, 2, commandBufferD3D12Desc, resourceSnapshot);

// Update state
if (!resourceSnapshot.restoreInitialState)
{
    for (size_t i = 0; i < resourceSnapshot.uniqueNum; i++)
    {
        // use "resourceSnapshot.unique[i].userArg" to get access to an app-side resource and update its state to "resourceSnapshot.unique[i].state"
    }
}

// IMPORTANT: NRD integration binds own descriptor pool and pipeline layout (root signature), don't forget to restore them if needed

//=======================================================================================================
// SHUTDOWN - DESTROY
//=======================================================================================================

NRD.Destroy();
```

Shader part:

```cpp
#include "NRD.hlsli"

// Pseudo code (this can be simplified for 1 path per pixel, see "NRD sample/simplex" branch)
Hit primaryHit; // aka 0 bounce, or PSR
Out out = (Out)0;

if (!OCCLUSION)
  out.specHitDist = NRD_FrontEnd_SpecHitDistAveraging_Begin();

for (int path = 0; path < pathNum; path++)
{
    float accumulatedHitDist = 0;
    float3 ray1 = 0;

    for (int bounce = 1; bounce <= bounceMaxNum; bounce++)
    {
        ...

        // Accumulate hit distance along the path (see NRD sample for the advanced approach)
        if (bounce == 1)
            accumulatedHitDist = hitDist;

        // Save sampling direction of the 1st bounce for SH denoisers
        if (bounce == 1 && SH)
            ray1 = ray;
    }

    // Normalize hit distances for REBLUR
    float normHitDist = accumulatedHitDist;
    if (REBLUR)
        normHitDist = REBLUR_FrontEnd_GetNormHitDist(accumulatedHitDist, primaryHit.viewZ, gHitDistSettings, isDiffusePath ? 1.0 : primaryHit.roughness);

    // Accumulate diffuse and specular separately for denoising
    if (isDiffusePath)
    {
        diffPathNum++;

        out.diffRadiance += Lsum;
        out.diffHitDist += normHitDist;

        if (SH)
            out.diffDirection += ray1;
    }
    else
    {
        out.specRadiance += Lsum;

        if (!OCCLUSION)
            NRD_FrontEnd_SpecHitDistAveraging_Add(out.specHitDist, normHitDist);
        else
          out.specHitDist += normHitDist;

        if (SH)
            out.specDirection += ray1;
    }
}

if (!OCCLUSION)
  NRD_FrontEnd_SpecHitDistAveraging_End(out.specHitDist);

// Radiance should already respect sampling probability => average across all paths
float invPathNum = 1.0 / float(pathNum);
out.diffRadiance *= invPathNum;
out.specRadiance *= invPathNum;

// Others must not include sampling probability => average only across diffuse / specular paths
float diffNorm = diffPathNum ? 1.0 / float( diffPathNum ) : 0.0;
out.diffHitDist *= diffNorm;
if (SH)
    result.diffDirection *= diffNorm;

float specNorm = diffPathNum < pathNum ? 1.0 / float( pathNum - diffPathNum ) : 0.0;
if (OCCLUSION)
  out.specHitDist *= specNorm;
if (SH)
    result.specDirection *= specNorm;

// Material de-modulation (convert irradiance into radiance)
float3 diffFactor, specFactor;
NRD_MaterialFactors(primaryHit.N, primaryHit.V, primaryHit.albedo, primaryHit.Rf0, primaryHit.roughness, diffFactor, specFactor);

out.diffRadiance /= diffFactor;
out.specRadiance /= specFactor;
```
</details>

Or alternatively, an app-side RHI or a native GAPI can be used explicitly:
* Create shaders from precompiled binary blobs
* Create an SRV for a texture (always `mip0`, no subresources)
* Create and bind 2 predefined samplers
* Invoke a Dispatch call (no raster, no VS/PS)
* Create 2D textures with SRV/UAV access

# INPUTS

[Non-noisy](#non-noisy-inputs) inputs (guides):
 - must not contain `NAN/INF` values

[Noisy](#noisy-inputs) inputs (signal to be denoised):
 - `NAN/INF` values are allowed outside of active viewport, i.e. `pixelPos >= CommonSettings::rectSize`
 - `NAN/INF` values are allowed outside of denoising range, i.e. `abs( viewZ ) >= CommonSettings::denoisingRange`

## NON-NOISY INPUTS

Commons inputs for primary hits (if *PSR* is not used, common use case) or for secondary hits (if *PSR* is used, valid only for 0-roughness):

* **IN\_MV** - non-jittered surface motion (`old = new + MV`)

  Modes:
  - *2D screen-space motion* - 2D motion doesn't provide information about movement along the view direction. *NRD* can reject history on dynamic objects in this case
  - *2.5D screen-space motion (recommended)* - similar to the 2D screen-space motion, but `.z = viewZprev - viewZ` (see [NRD sample/GetMotion](https://github.com/NVIDIA-RTX/NRD-Sample/blob/9deb12a5408c4e2e07a6ff261f0a1051dd22f5d6/Shaders/Include/Shared.hlsli#L358))
  - *3D world-space motion* - camera motion should not be included (it's already in the matrices). In other words, if there are no moving objects, all motion vectors must be `0` even if the camera is moving

  Motion vector scaling can be provided via `CommonSettings::motionVectorScale`. *NRD* expectations:
  - Use `CommonSettings::isMotionVectorInWorldSpace = true` for 3D world-space motion
  - Use `CommonSettings::isMotionVectorInWorldSpace = false` and `CommonSettings::motionVectorScale[2] == 0` for 2D screen-space motion
  - Use `CommonSettings::isMotionVectorInWorldSpace = false` and `CommonSettings::motionVectorScale[2] != 0` for 2.5D screen-space motion

* **IN\_NORMAL\_ROUGHNESS** - surface world-space normal and *linear* roughness

  Normal and roughness encoding must be controlled via *Cmake* parameters `NRD_NORMAL_ENCODING` and `NRD_ROUGHNESS_ENCODING`. Encoding settings can be known at runtime by accessing `LibraryDesc::normalEncoding` and `LibraryDesc::roghnessEncoding` respectively. `NormalEncoding` and `RoughnessEncoding` enums briefly describe encoding variants. It's recommended to use `NRD.hlsli/NRD_FrontEnd_PackNormalAndRoughness` to match decoding.

  *NRD* computes local curvature using provided normals. Less accurate normals can lead to banding in curvature and local flatness. `RGBA8` normals is a good baseline, but `R10G10B10A10` oct-packed normals improve curvature calculations and specular tracking as the result.

  If `materialID` is provided and supported by encoding, *NRD* diffuse and specular denoisers won't mix up surfaces with different material IDs.

* **IN\_VIEWZ** - view-space Z coordinate of primary hits (linearized g-buffer depth)

  Positive and negative values are supported. Z values in all pixels must be in the same space, matching space defined by matrices passed to NRD. If, for example, the protagonist's hands are rendered using special matrices, Z values should be computed as:
  - reconstruct world position using special matrices for "hands"
  - project on screen using matrices passed to NRD
  - `.w` component is positive view Z (or just transform world-space position to main view space and take `.z` component)

* **IN\_DIFF/SPEC\_CONFIDENCE** - convidence in the accumulated history represented in `[0; 1]` range

  These inputs are optional and are used only if `CommonSettings::isHistoryConfidenceAvailable = true`. *REBLUR* and *RELAX* have embedded anti-lag techniques, but if properly computed, using confidence inputs is the best way to mitigate temporal lags. They are easy and cheap to compute. See this [section](#history-confidence) for more details.

The illustration below shows expected inputs for a primary hit `A`:

![Input without PSR](Images/InputsWithoutPsr.png)

```cpp
hitDistance = length( B - A ); // hitT for 1st bounce (recommended baseline)

IN_VIEWZ = TransformToViewSpace( A ).z;
IN_NORMAL_ROUGHNESS = GetNormalAndRoughnessAt( A );
IN_MV = GetMotionAt( A );
```

See `NRDDescs.h` and `NRD.hlsli` for more details and descriptions of other inputs and outputs. See this [section](#interaction-with-primary-surface-replacements) for interaction with Primary Surface Replacements (PSRs).

## NOISY INPUTS

NRD sample is a good start to familiarize yourself with input requirements and best practices, but main requirements can be summarized to:

Radiance:
- Since *NRD* denoisers accumulate signals for a limited number of frames, the input signal must converge *reasonably* well for this number of frames. `REFERENCE` denoiser can be used to estimate temporal signal quality
- Since *NRD* denoisers process signals spatially, high-energy fireflies in the input signal should be avoided. Some of them can be removed by enabling anti-firefly filter in *NRD*, but it will only work if the "background" signal is confident. The worst case is having a single pixel with a high energy divided by a very small PDF to represent the lack of energy in neighboring non-representative (black) pixels. Probabilistic diffuse / specular split for the 1st bounce requires special treatment described in `HitDistanceReconstructionMode`. In case of probabilistic split for 2nd+ bounces, it's still recommended to clamp diffuse / specular probabilities to a sane range to avoid division by a very small value, leading to a high energy firefly, difficult to get rid of in a short amount of time. Energy increase should not be more than 20x-30x, what corresponds to around `0.05` min probability. `0` and `1` probabilities are absolutely acceptable (for example, metals don't have diffuse component)
- Radiance must be separated into diffuse and specular at primary hit (or secondary hit in case of *PSR*)

Hit distance (*REBLUR* and *RELAX*):
- `hitT` can't be negative
- `hitT` must not include primary hit distance
- `hitT` for the first bounce after the primary hit or *PSR* must be provided "as is"
- `hitT` for subsequent bounces and for bounces before *PSR* must be adjusted by curvature and lobe energy dissipation on the application side
  - Do not pass *sum of lengths of all segments* as `hitT`. A solid baseline is to use hit distance for the 1st bounce only, it works well for diffuse and specular signals
  - *NRD sample* uses more complex approach for accumulating `hitT` along the path, which takes into account energy dissipation due to lobe spread and curvature at the current hit
- For rays pointing inside the surface (VNDF sampling can easily produce those), `hitT` must be set to 0 (but better to not cast such rays)
- Noise in hit distances must follow a diffuse or specular lobe. It implies that `hitT` for `roughness = 0` must be clean (if probabilistic sampling is not in use)
- In case of probabilistic diffuse / specular selection at the primary hit, provided `hitT` must follow the following rules:
  - Should not be divided by `PDF`
  - If diffuse or specular sampling is skipped, `hitT` must be set to `0` for corresponding signal type
  - `hitDistanceReconstructionMode` must be set to something other than `OFF`, but bear in mind that the search area is limited to 3x3 (or 5x5). In other words, it's the application's responsibility to guarantee a valid sample in this area. It can be achieved by clamping probabilities and using Bayer-like dithering (see [NRD sample/clamping lobe selection probability](https://github.com/NVIDIA-RTX/NRD-Sample/blob/6f1a294333dd32dd5ea404845354d76315824add/Shaders/TraceOpaque.cs.hlsl#L223))
  - Pre-pass must be enabled (i.e. `diffusePrepassBlurRadius` and `specularPrepassBlurRadius` must be set to 20-70 pixels) to compensate entropy increase, since radiance in valid samples is divided by probability to compensate 0 values in some neighbors
- Probabilistic split for 2nd+ bounces is absolutely acceptable
- In case of many paths per pixel `hitT` for specular must be "averaged" by `NRD.hlsli/NRD_FrontEnd_SpecHitDistAveraging_*` functions
- For *REBLUR* hits distance must be normalized using `NRD.hlsli/REBLUR_FrontEnd_GetNormHitDist`

Distance to occluder (*SIGMA*):
- visibility ray must be cast from the point of interest to a light source ( i.e. *not* from a light source )
- `ACCEPT_FIRST_HIT_AND_END_SEARCH` ray flag can't be used to optimize tracing, because it can lead to wrong potentially very long hit distances from random distant occluders
- `hit` means "occluder is hit"
- `miss` means "light is hit"
- `NoL <= 0` - 0 (it's very important!)
- `NoL > 0, hit` - hit distance
- `NoL > 0, miss` - >= NRD_FP16_MAX

See `NRDDescs.h` and `NRD.hlsli` for more details and descriptions of other inputs and outputs.

# RECOMMENDATIONS AND BEST PRACTICES

Denoising is not a panacea or miracle. Denoising works best with ray tracing results produced by a suitable form of importance sampling. Additionally, *NRD* has its own restrictions. The following suggestions should help to achieve best image quality:

## VALIDATION LAYER

![Validation](Images/Validation.png)

If `CommonSettings::enableValidation = true` *REBLUR* & *RELAX* denoisers render debug information into `OUT_VALIDATION` output. Alpha channel contains layer transparency to allow easy mix with the final image on the application side. The following viewport layout is used on the screen:

| 0 | 1 | 2 | 3 |
|---|---|---|---|
| 4 | 5 | 6 | 7 |
| 8 | 9 | 10| 11|
| 12| 13| 14| 15|

where:

- Viewport 0 - world-space normals
- Viewport 1 - linear roughness
- Viewport 2 - linear viewZ
  - green = `+`
  - blue = `-`
  - red = `out of denoising range`
- Viewport 3 - difference between MVs, coming from `IN_MV`, and expected MVs, assuming that the scene is static
  - blue = `out of screen`
  - pixels with moving objects have non-0 values
- Viewport 4 - world-space grid & camera jitter:
  - 1 cube = `1 unit`
  - the square in the bottom-right corner represents a pixel with accumulated samples
  - the red boundary of the square marks jittering outside of the pixel area
- Viewport 7 - amount of virtual history
- Viewport 8 - number of accumulated frames for diffuse signal (checkerboared red = `history reset`)
- Viewport 11 - number of accumulated frames for specular signal (checkerboared red = `history reset`)
- Viewport 12 - input normalized `hitT` for diffuse signal (ambient occlusion, AO)
- Viewport 15 - input normalized `hitT` for specular signal (specular occlusion, SO)

## MATERIAL DEMODULATION

*NRD* has been designed to work with pure radiance coming from a particular direction. This means that data in the form "something / probability" should be avoided if possible because overall entropy of the input signal will be increased (but it doesn't mean that denoising won't work). Additionally, it means that materials needs to be decoupled from the input signal, i.e. *irradiance*, typically produced by a path tracer, needs to be transformed into *radiance*, i.e. BRDF should be applied **after** denoising. This is achieved by using "demodulation":

    // Diffuse
    Denoising( diffuseRadiance * albedo ) → NRD( diffuseRadiance / albedo ) * albedo

    // Specular
    float3 preintegratedBRDF = PreintegratedBRDF( Rf0, N, V, roughness )
    Denoising( specularRadiance * BRDF ) → NRD( specularRadiance * BRDF / preintegratedBRDF ) * preintegratedBRDF

Use `NRD.hlsli/NRD_MaterialFactors` helper to compute material demodulation factors.

## INTERACTION WITH PRIMARY SURFACE REPLACEMENTS

When denoising reflections in pure mirrors, some advantages can be reached if *NRD* "sees" the first "non-pure mirror" point after a series of pure mirror bounces (delta events). This point is called [*Primary Surface Replacement (PSR)*](https://developer.nvidia.com/blog/rendering-perfect-reflections-and-refractions-in-path-traced-games/).

Notes, requirements and restrictions:
- the primary hit (0th bounce) gets replaced with the first "non-pure mirror" hit in the bounce chain - this hit becomes *PSR*
- all associated data in the g-buffer gets replaced by *PSR* data
- the camera "sees" PSR like the mirror surface in-between doesn't exist. This space is called virtual world space
  - virtual space position lies on the same view vector as the primary hit position, but the position is elongated. Elongation depends on `hitT` and curvature at hits, starting from the primary hit
  - virtual space normal is the normal at *PSR* hit mirrored several times  in the reversed order until the primary hit is reached
- *PSR* data is NOT always data at the *PSR* hit!
  - material properties (albedo, metalness, roughness etc.) are from *PSR* hit
  - `IN_NORMAL_ROUGHNESS` contains normal at virtual world space and roughness at *PSR*
  - `IN_VIEWZ` contains `viewZ` of the virtual position, potentially adjusted several times by curvature at hits
  - `IN_MV` contains motion of the virtual position, potentially adjusted several times by curvature at hits
  - accumulated `hitT` starts at the *PSR* hit, potentially adjusted several times by curvature at hits
  - curvature should be taken into account starting from the 1st bounce, because the primary surface normal will be replaced by *PSR* normal, i.e. the former will be unreachable on the *NRD* side
  - ray direction for *NRD* must be transformed into virtual space

IMPORTANT: in other words, *PSR* is perfect for flat mirrors. *PSR* on curved surfaces works even without respecting curvature, but reprojection artefacts can appear.

In case of *PSR* *NRD* disocclusion logic doesn't take curvature at primary hit into account, because data for primary hits is replaced. This can lead to more intense disocclusions on bumpy surfaces due to significant ray divergence. To mitigate this problem 2x-10x larger `CommonSettings::disocclusionThreshold` can be used. This is an applicable solution if the denoiser is used to denoise surfaces with *PSR* only (glass only, for example). In a general case, when *PSR* and normal surfaces are mixed on the screen, higher disocclusion thresholds are needed only for pixels with *PSR*. This can be achieved by using `IN_DISOCCLUSION_THRESHOLD_MIX` input to smoothly mix baseline `CommonSettings::disocclusionThreshold` into bigger `CommonSettings::disocclusionThresholdAlternate`. Most likely the increased disocclusion threshold is needed only for pixels with normal details at primary hits (local curvature is not zero).

The illustration below shows expected inputs for primary hit `A` replaced with hit `B` (*PSR*):

![Input with PSR](Images/InputsWithPsr.png)

```cpp
hitDistance = length( C - B ); // hitT for 2nd bounce, but it's 1st bounce in the reflected world
Bvirtual = A + viewVector * length( B - A );

IN_VIEWZ = TransformToViewSpace( Bvirtual ).z;
IN_NORMAL_ROUGHNESS = GetVirtualSpaceNormalAndRoughnessAt( B );
IN_MV = GetMotionAt( B );
```

## IMPROVING OUTPUT QUALITY

The temporal part of *NRD* naturally suppresses jitter, which is essential for upscaling techniques. If an *SH* denoiser is in use, a high quality resolve can be applied to the final output to regain back macro details, micro details and per-pixel jittering. As an example, the image below demonstrates the results *after* and *before* resolve with active *DLSS* (quality mode).

![Resolve](Images/Resolve.jpg)

The resolve process takes place on the application side and has the following modular structure:
- construct an SG (spherical gaussian) light
- apply diffuse or specular resolve function to reconstruct macro details
- apply re-jittering to reconstruct micro details
- (optionally) or just extract unresolved color (fully matches the output of a corresponding non-SH denoiser)

Re-jittering math with minorly modified inputs can also be used with RESTIR produced sampling without involving SH denoisers. You only need to get light direction in the current pixel from RESTIR. Despite that RESTIR produces noisy light selections, its low variations can be easily handled by DLSS or other upscaling techs.

<details>
<summary>Shader code (CLICK):</summary>

```cpp
// Diffuse
float4 diff = gIn_Diff[ pixelPos ];
float4 diff1 = gIn_DiffSh[ pixelPos ];

NRD_SG diffSg = REBLUR_BackEnd_UnpackSh( diff, diff1 );

// Specular
float4 spec = gIn_Spec[ pixelPos ];
float4 spec1 = gIn_SpecSh[ pixelPos ];

NRD_SG specSg = REBLUR_BackEnd_UnpackSh( spec, spec1 );

// ( Optional ) AO / SO ( available only for REBLUR )
diff.w = diffSg.normHitDist;
spec.w = specSg.normHitDist;

if( gResolve )
{
    // ( Optional ) replace "roughness" with "roughnessAA"
    roughness = NRD_SG_ExtractRoughnessAA( specSg );

    // Regain macro-details
    diff.xyz = NRD_SG_ResolveDiffuse( diffSg, N ); // or NRD_SH_ResolveDiffuse( sg, N )
    spec.xyz = NRD_SG_ResolveSpecular( specSg, N, V, roughness );

    // Regain micro-details & jittering // TODO: preload N and Z into SMEM
    float3 Ne = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ pixelPos + int2( 1, 0 ) ] ).xyz;
    float3 Nw = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ pixelPos + int2( -1, 0 ) ] ).xyz;
    float3 Nn = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ pixelPos + int2( 0, 1 ) ] ).xyz;
    float3 Ns = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ pixelPos + int2( 0, -1 ) ] ).xyz;

    float Ze = gIn_ViewZ[ pixelPos + int2( 1, 0 ) ];
    float Zw = gIn_ViewZ[ pixelPos + int2( -1, 0 ) ];
    float Zn = gIn_ViewZ[ pixelPos + int2( 0, 1 ) ];
    float Zs = gIn_ViewZ[ pixelPos + int2( 0, -1 ) ];

    float2 scale = NRD_SG_ReJitter( diffSg, specSg, Rf0, V, roughness, viewZ, Ze, Zw, Zn, Zs, N, Ne, Nw, Nn, Ns );

    diff.xyz *= scale.x;
    spec.xyz *= scale.y;
}
else
{
    // ( Optional ) Unresolved color matching the non-SH version of the denoiser
    diff.xyz = NRD_SG_ExtractColor( diffSg );
    spec.xyz = NRD_SG_ExtractColor( specSg );
}
```

</details>

## HISTORY CONFIDENCE

User-provided history confidence inputs (`IN_DIFF_CONFIDENCE` and `IN_SPEC_CONFIDENCE`) are essential to preserve the responsiviness of the denoised output.

## HAIR DENOISING TIPS

*NRD* tries to preserve jittering at least on geometrical edges, it's essential for upscalers, which are usually applied at the end of the rendering pipeline. It naturally moves the problem of anti-aliasing to the application side. In order, it implies the following obvious suggestions:
- trace at higher resolution, denoise, apply AA and downscale
- apply a high-quality upscaler in "AA-only" mode, i.e. without reducing the tracing resolution (for example, *DLSS* in *DLAA mode*)

Sub-pixel thin geometry of strand-based hair transforms "normals guide" into jittering & flickering pixel mess, i.e. the guide itself becomes noisy. It worsens denoising IQ. At least for *NRD* better to replace geometry normals in "normals guide" with a vector `= normalize( cross( T, B ) )`, where:
- `T` - hair strand tangent vector
- `B` - is not a classic binormal, it's more an averaged direction to a bunch of closest hair strands (in many cases it's a binormal vector of underlying head / body mesh)
  - `B` can be simplified to `normalize( cross( V, T ) )`, where `V` is the view vector
  - in other words, `B` must follow the following rules:
    - `cross( T, B ) != 0`
    - `B` must not follow hair strand "tube"
- search for `FLAG_HAIR` in [NRD sample](https://github.com/NVIDIA-RTX/NRD-Sample) for more details

Hair strands tangent vectors *can't* be used as "normals guide" for *NRD* due to BRDF and curvature related calculations, requiring a vector, which can be considered a "normal" vector.

## COMBINED DENOISING OF DIRECT AND INDIRECT LIGHTING

Denoising process is driven by hit distances (with the exception that *RELAX* uses hit distances only in the Pre-Pass and for specular tracking). Denoising of combined direct and indirect lighting implies mixing corresponding hit distances into one value for NRD. Here are some suggestions:

1. For specular signal use indirect `hitT` for both direct and indirect lighting

The reason is that the denoiser uses `hitT` mostly for calculating motion vectors for reflections. For that purpose, the denoiser expects to see `hitT` from surfaces that are in the specular reflection lobe. When calculating direct lighting (*NEE/RTXDI*), we select a light per pixel, and the distance to that light becomes the `hitT` for both diffuse and specular channels. In many cases, the light is selected for a surface because of its diffuse contribution, not specular, which makes the specular channel contain the `hitT` of a diffuse light. That confuses the denoiser and breaks reprojection. On the other hand, the indirect specular `hitT` is always computed by tracing rays in the specular lobe.

2. For diffuse signal hit distance can be adjusted by mixing `hitT` from direct and indirect rays to get sharper shadows

Use 1st bounce hit distance for the indirect lighting in the pseudo-code below:
```cpp
float directHitDistContribution = directDiffuseLuminance / ( directDiffuseLuminance + indirectDiffuseLuminance + EPS );

const float maxContribution = 0.5; // this is adjustable
directHitDistContribution = min( directHitDistContribution, maxContribution ); // avoid over-sharpening

float hitDist = lerp( indirectDiffuseHitDist, directDiffuseHitDist, directHitDistContribution );
```

## INTERACTION WITH FRAME GENERATION TECHNIQUES

Frame generation (FG) techniques boost FPS by interpolating between 2 last available frames. *NRD* works better when frame rate increases, because it gets more data per second. It's not the case for FG, because all rendering pipeline underlying passes (like, denoising) continue to work on the original non-boosted framerate. `GetMaxAccumulatedFrameNum` helper should get a real FPS, not a fake one.

## OTHER

**[NRD]** All denoising and path-tracing best practices are in *NRD sample*.

**[NRD]** Use "debug" *NRD* during development, it has many useful debug checks saving from common pitfalls.

**[NRD]** Read all comments in `NRDDescs.h`, `NRDSettings.h` and `NRD.hlsli`.

**[NRD]** The *NRD API* has been designed to support integration into native VULKAN apps. If the RHI you work with is DX11-like, not all provided data will be needed. `NRDIntegration.hpp` can be used as a guide demonstrating how to map NRD API to a Vulkan-like RHI.

**[NRD]** *NRD* requires linear roughness and world-space normals. See `NRD.hlsli` for more details and supported customizations.

**[NRD]** *NRD* requires non-jittered matrices.

**[NRD]** Most denoisers do not write into output pixels outside of `CommonSettings::denoisingRange`. A hack - if there are areas (besides sky), which don't require denoising (for example, casting a specular ray only if roughness is less than some threshold), providing `viewZ > CommonSettings::denoisingRange` in **IN\_VIEWZ** texture for such pixels will effectively skip denoising. Additionally, the data in such areas won't contribute to the final result.

**[NRD]** When upgrading to the latest version keep an eye on `ResourceType` enumeration. The order of the input slots can be changed or something can be added, you need to adjust the inputs accordingly to match the mapping. Or use *NRD integration* to simplify the process.

**[NRD]** Functions `NRD.hlsli/XXX_FrontEnd_PackRadianceAndHitDist` perform optional `NAN/INF` clearing of the input signal. There is a boolean to skip these checks.

**[NRD]** All denoisers work with positive RGB inputs (some denoisers can change color space in *front end* functions). For better image quality, HDR color inputs need to be in a sane range [0; 250], because the internal pipeline uses FP16 and *RELAX* tracks second moments of the input signal, i.e. `x^2` must fit into FP16 range. If the color input is in a wider range, any form of non-aggressive color compression can be applied (linear scaling, pow-based or log-based methods). *REBLUR* supports wider HDR ranges, because it doesn't track second moments. Passing pre-exposured colors (i.e. `color * exposure`) is not recommended, because a significant momentary change in exposure is hard to react to in this case.

**[NRD]** *NRD* can track camera motion internally. For the first time pass all MVs set to 0 (you can use `CommonSettings::motionVectorScale = {0}` for this) and set `CommonSettings::isMotionVectorInWorldSpace = true`, it will allow you to simplify the initial integration. Enable application-provided MVs after getting denoising working on static objects.

**[NRD]** Using 2D MVs can lead to massive history reset on moving objects, because 2D motion provides information only about pixel screen position but not about real 3D world position. Consider using 2.5D or 3D MVs instead. 2.5D motion, which is 2D motion with additionally provided `viewZ` delta (i.e. `viewZprev = viewZ + MV.z`), is even better, because it has the same benefits as 3D motion, but doesn't suffer from imprecision problems caused by world-space delta rounding to FP16 during MV patching on the *NRD* side.

**[NRD]** Firstly, try to get a working reprojection on a diffuse signal for camera rotations only (without camera motion).

**[NRD]** Diffuse and specular signals must be separated at primary hit (or at secondary hit in case of *PSR*).

**[NRD]** Denoising logic is driven by provided hit distances. For indirect lighting denoising passing hit distance for the 1st bounce only is a good baseline. For direct lighting a distance to an occluder or a light source is needed. Primary hit distance must be excluded in any case.

**[NRD]** Importance sampling is recommended to achieve good results in case of complex lighting environments. Consider using:
   - Cosine distribution for diffuse from non-local light sources
   - VNDF sampling for specular
   - Custom importance sampling for local light sources (*RTXDI*).

**[NRD]** Any form of a radiance cache (*[SHARC](https://github.com/NVIDIA-RTX/SHARC)* or *[NRC](https://github.com/NVIDIA-RTX/NRC)*) is highly recommended to achieve better signal quality and improve behavior in disocclusions.

**[NRD]** Additionally the quality of the input signal can be increased by re-using already denoised information from the current or the previous frame.

**[NRD]** Hit distances should come from an importance sampling method. But if denoising of AO/SO is needed, AO/SO must come from cos-weighted (or VNDF) sampling in a tradeoff of IQ.

**[NRD]** Low discrepancy sampling (blue noise) helps to get more stable output in 0.5-1 rpp mode. It's a must for REBLUR-based Ambient and Specular Occlusion denoisers and SIGMA.

**[NRD]** It's recommended to set `CommonSettings::accumulationMode` to `RESTART` for a single frame, if a history reset is needed. If history buffers are recreated or contain garbage, it's recommended to use `CLEAR_AND_RESTART` for a single frame. `CLEAR_AND_RESTART` is not free because clearing is done in a compute shader. Render target clears on the application side should be prioritized over this solution, if possible.

**[NRD]** If normal-roughness encoding supports `materialID`, the following features become available:
- `CommonSettings::minMaterialForDiffuse, minMaterialForSpecular` - `materialID` comparison, useful to not mix diffuse between dielectrics (non-0 diffuse) and metals (0 diffuse)
- `CommonSettings::strandMaterialID` - marks hair (grass) geometry to enable "under-the-hood" tweaks
- `CommonSettings::cameraAttachedReflectionMaterialID` - marks reflections of camera attached objects

**[NRD]** If you are unsure of which denoiser settings to use - use defaults via `{}` construction. It helps to improve compatibility with future versions and offers optimal IQ, because default settings are always adjusted by recent algorithmic changes.

**[NRD]** Input signal quality can be improved by enabling *pre-pass* via setting `diffusePrepassBlurRadius` and `specularPrepassBlurRadius` to a non-zero value. Pre-pass is needed more for specular and less for diffuse, because pre-pass outputs optimal hit distance for specular tracking. For relatively clean signals *pre-pass* may introduce additional blur. In this case `ReblurSettings::usePrepassOnlyForSpecularMotionEstimation = true` can be used in conjunction with `diffusePrepassBlurRadius = 0`.

**[NRD]** In case of probabilistic diffuse / specular split at the primary hit, hit distance reconstruction pass must be enabled, if exposed in the denoiser (see `HitDistanceReconstructionMode`).

**[NRD]** In case of probabilistic diffuse / specular split at the primary hit, pre-pass must be enabled, if exposed in the denoiser (see `diffusePrepassBlurRadius` and `specularPrepassBlurRadius`).

**[NRD]** Maximum number of accumulated frames can be FPS dependent. The following formula can be used on the application side to adjust `maxAccumulatedFrameNum`, `maxFastAccumulatedFrameNum` and potentially `historyFixFrameNum` too:
```
maxAccumulatedFrameNum = accumulationPeriodInSeconds * FPS
```

**[NRD]** Fast history is the input signal, accumulated for a few frames. Fast history helps to minimize lags in the main history, which is accumulated for more frames. The number of accumulated frames in the fast history needs to be carefully tuned to avoid introducing significant bias and dirt. Initial integration should be done with default settings. Bear in mind the following recommendation:
```
maxAccumulatedFrameNum > maxFastAccumulatedFrameNum > historyFixFrameNum
```

**[NRD]** In case of quarter resolution tracing and denoising use `pixelPos / 2` as texture coordinates. Using a "rotated grid" approach (when a pixel gets selected from 2x2 footprint one by one) is not recommended because it significantly bumps entropy of non-noisy inputs, leading to more disocclusions. In case of *REBLUR* it's recommended to increase `sigmaScale` in antilag settings. "Nearest Z" upsampling works best for upscaling of the denoised output. Code, as well as upsampling function, can be found in *NRD sample* releases before 3.10.

**[NRD]** *SH* denoisers can use more relaxed `lobeAngleFraction`. It can help to improve stability, while details will be reconstructed back by *SG* resolve.

**[REBLUR]** If more performance is needed, consider using `REBLUR_PERFORMANCE_MODE = ON`.

**[REBLUR]** *REBLUR* expects hit distances in a normalized form. To avoid mismatching, `NRD.hlsli/REBLUR_FrontEnd_GetNormHitDist` must be used for normalization. Normalization parameters should be passed into *NRD* as `HitDistanceParameters` for internal hit distance denormalization. Some tweaking can be needed here, but in most cases default `HitDistanceParameters` works well. *REBLUR* outputs denoised normalized hit distance, which can be used by the application as ambient or specular occlusion (AO & SO) (see unpacking functions from `NRD.hlsli`).

**[REBLUR/RELAX]** Antilag parameters need to be carefully tuned. Initial integration should be done with disabled antilag.

**[RELAX]** *RELAX* works well with signals produced by *RTXDI* or very clean high RPP signals. The Sweet Home of *RELAX* is *RTXDI* sample.

**[SIGMA]** Using "blue" noise helps to minimize shadow shimmering and flickering. It works best if the pattern has limited number of animated frames (4-8) or it is static on the screen.

**[SIGMA]** *SIGMA* can be used for multi-light shadow denoising if applied "per light". `maxStabilizedFrameNum` can be set to `0` to disable temporal history. It provides the following benefits:
 - light count independent memory usage
 - no need to manage history buffers for lights

**[SIGMA]** In theory *SIGMA_TRANSLUCENT_SHADOW* can be used as a "single-pass" shadow denoiser for shadows from multiple light sources:

*L[i]* - unshadowed analytical lighting from a single light source (**not noisy**)<br/>
*S[i]* - stochastically sampled light visibility for *L[i]* (**noisy**)<br/>
*&Sigma;( L[i] )* - unshadowed analytical lighting, typically a result of tiled lighting (HDR, not in range [0; 1])<br/>
*&Sigma;( L[i] &times; S[i] )* - final lighting (what we need to get)

The idea:<br/>
*L1 &times; S1 + L2 &times; S2 + L3 &times; S3 = ( L1 + L2 + L3 ) &times; [ ( L1 &times; S1 + L2 &times; S2 + L3 &times; S3 ) / ( L1 + L2 + L3 ) ]*

Or:<br/>
*&Sigma;( L[i] &times; S[i] ) = &Sigma;( L[i] ) &times; [ &Sigma;( L[i] &times; S[i] ) / &Sigma;( L[i] ) ]*<br/>
*&Sigma;( L[i] &times; S[i] ) / &Sigma;( L[i] )* - normalized weighted sum, i.e. pseudo translucency (LDR, in range [0; 1])

Input data preparation example:
```cpp
float3 Lsum = 0;
float3 LSsum = 0.0;
float Wsum = 0.0;
float Psum = 0.0;

for( uint i = 0; i < N; i++ )
{
    float3 L = ComputeLighting( i );
    Lsum += L;

    // "distanceToOccluder" should respect rules described in NRD.hlsli in "INPUT PARAMETERS" section
    float distanceToOccluder = SampleShadow( i );
    float shadow = !IsOccluded( distanceToOccluder );
    LSsum += L * shadow;

    // The weight should be zero if a pixel is not in the penumbra, but it is not trivial to compute...
    float weight = ...;
    weight *= Luminance( L );
    Wsum += weight;

    float penumbraRadius = SIGMA_FrontEnd_PackPenumbra( ... ).x;
    Psum += penumbraRadius * weight;
}

float3 translucency = LSsum / max( Lsum, NRD_EPS );
float penumbraRadius = Psum / max( Wsum, NRD_EPS );
```

After denoising the final result can be computed as:

*&Sigma;( L[i] &times; S[i] )* = *&Sigma;( L[i] )* &times; *OUT_SHADOW_TRANSLUCENCY.yzw*

Is this a biased solution? If spatial filtering is off - no, because we just reorganized the math equation. If spatial filtering is on - yes, because denoising will be driven by most important light in a given pixel.

**This solution is limited** and hard to use:
- obviously, can be used "as is" if shadows don't overlap (*weight* = 1)
- if shadows overlap, a separate pass is needed to analyze noisy input and classify pixels as *umbra* - *penumbra* (and optionally *empty space*). Raster shadow maps can be used for this if available
- it is not recommended to mix 1 cd and 100000 cd lights, since FP32 texture will be needed for a weighted sum.
In this case, it's better to process the sun and other bright light sources separately.

# MEMORY USAGE

The *Persistent* column (matches *NRD Permanent pool*) indicates how much of the *Working set* is required to be left intact for subsequent frames of the application. This memory stores the history resources consumed by NRD. The *Aliasable* column (matches *NRD Transient pool*) shows how much of the *Working set* may be aliased by textures or other resources used by the application outside of the operating boundaries of NRD.

| Resolution |                             Denoiser | Working set (Mb) |  Persistent (Mb) |   Aliasable (Mb) |
|------------|--------------------------------------|------------------|------------------|------------------|
|      1080p |                       REBLUR_DIFFUSE |            76.19 |            50.75 |            25.44 |
|            |             REBLUR_DIFFUSE_OCCLUSION |            36.06 |            27.50 |             8.56 |
|            |                    REBLUR_DIFFUSE_SH |           109.94 |            67.62 |            42.31 |
|            |                      REBLUR_SPECULAR |            95.25 |            59.25 |            36.00 |
|            |            REBLUR_SPECULAR_OCCLUSION |            44.56 |            36.00 |             8.56 |
|            |                   REBLUR_SPECULAR_SH |           129.00 |            76.12 |            52.88 |
|            |              REBLUR_DIFFUSE_SPECULAR |           148.12 |            88.88 |            59.25 |
|            |    REBLUR_DIFFUSE_SPECULAR_OCCLUSION |            59.44 |            42.38 |            17.06 |
|            |           REBLUR_DIFFUSE_SPECULAR_SH |           232.50 |           122.62 |           109.88 |
|            | REBLUR_DIFFUSE_DIRECTIONAL_OCCLUSION |            71.94 |            48.62 |            23.31 |
|            |                        RELAX_DIFFUSE |            90.81 |            54.88 |            35.94 |
|            |                     RELAX_DIFFUSE_SH |           158.31 |            88.62 |            69.69 |
|            |                       RELAX_SPECULAR |           101.44 |            63.38 |            38.06 |
|            |                    RELAX_SPECULAR_SH |           168.94 |            97.12 |            71.81 |
|            |               RELAX_DIFFUSE_SPECULAR |           168.94 |            97.12 |            71.81 |
|            |            RELAX_DIFFUSE_SPECULAR_SH |           303.94 |           164.62 |           139.31 |
|            |                         SIGMA_SHADOW |            31.88 |             8.44 |            23.44 |
|            |            SIGMA_SHADOW_TRANSLUCENCY |            50.81 |             8.44 |            42.38 |
|            |                            REFERENCE |            33.75 |            33.75 |             0.00 |
|            |                                      |                  |                  |                  |
|      1440p |                       REBLUR_DIFFUSE |           135.06 |            90.00 |            45.06 |
|            |             REBLUR_DIFFUSE_OCCLUSION |            63.81 |            48.75 |            15.06 |
|            |                    REBLUR_DIFFUSE_SH |           195.06 |           120.00 |            75.06 |
|            |                      REBLUR_SPECULAR |           168.81 |           105.00 |            63.81 |
|            |            REBLUR_SPECULAR_OCCLUSION |            78.81 |            63.75 |            15.06 |
|            |                   REBLUR_SPECULAR_SH |           228.81 |           135.00 |            93.81 |
|            |              REBLUR_DIFFUSE_SPECULAR |           262.56 |           157.50 |           105.06 |
|            |    REBLUR_DIFFUSE_SPECULAR_OCCLUSION |           105.06 |            75.00 |            30.06 |
|            |           REBLUR_DIFFUSE_SPECULAR_SH |           412.56 |           217.50 |           195.06 |
|            | REBLUR_DIFFUSE_DIRECTIONAL_OCCLUSION |           127.56 |            86.25 |            41.31 |
|            |                        RELAX_DIFFUSE |           161.31 |            97.50 |            63.81 |
|            |                     RELAX_DIFFUSE_SH |           281.31 |           157.50 |           123.81 |
|            |                       RELAX_SPECULAR |           180.06 |           112.50 |            67.56 |
|            |                    RELAX_SPECULAR_SH |           300.06 |           172.50 |           127.56 |
|            |               RELAX_DIFFUSE_SPECULAR |           300.06 |           172.50 |           127.56 |
|            |            RELAX_DIFFUSE_SPECULAR_SH |           540.06 |           292.50 |           247.56 |
|            |                         SIGMA_SHADOW |            56.38 |            15.00 |            41.38 |
|            |            SIGMA_SHADOW_TRANSLUCENCY |            90.12 |            15.00 |            75.12 |
|            |                            REFERENCE |            60.00 |            60.00 |             0.00 |
|            |                                      |                  |                  |                  |
|      2160p |                       REBLUR_DIFFUSE |           287.00 |           191.25 |            95.75 |
|            |             REBLUR_DIFFUSE_OCCLUSION |           135.62 |           103.62 |            32.00 |
|            |                    REBLUR_DIFFUSE_SH |           414.50 |           255.00 |           159.50 |
|            |                      REBLUR_SPECULAR |           358.69 |           223.12 |           135.56 |
|            |            REBLUR_SPECULAR_OCCLUSION |           167.50 |           135.50 |            32.00 |
|            |                   REBLUR_SPECULAR_SH |           486.19 |           286.88 |           199.31 |
|            |              REBLUR_DIFFUSE_SPECULAR |           557.88 |           334.69 |           223.19 |
|            |    REBLUR_DIFFUSE_SPECULAR_OCCLUSION |           223.31 |           159.44 |            63.88 |
|            |           REBLUR_DIFFUSE_SPECULAR_SH |           876.62 |           462.19 |           414.44 |
|            | REBLUR_DIFFUSE_DIRECTIONAL_OCCLUSION |           271.12 |           183.31 |            87.81 |
|            |                        RELAX_DIFFUSE |           342.81 |           207.25 |           135.56 |
|            |                     RELAX_DIFFUSE_SH |           597.81 |           334.75 |           263.06 |
|            |                       RELAX_SPECULAR |           382.69 |           239.12 |           143.56 |
|            |                    RELAX_SPECULAR_SH |           637.69 |           366.62 |           271.06 |
|            |               RELAX_DIFFUSE_SPECULAR |           637.69 |           366.62 |           271.06 |
|            |            RELAX_DIFFUSE_SPECULAR_SH |          1147.69 |           621.62 |           526.06 |
|            |                         SIGMA_SHADOW |           119.94 |            31.88 |            88.06 |
|            |            SIGMA_SHADOW_TRANSLUCENCY |           191.56 |            31.88 |           159.69 |
|            |                            REFERENCE |           127.50 |           127.50 |             0.00 |
