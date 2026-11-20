---
name: nrd-integration-review
description: Review NRD integrations against the installed API, README, public headers, and mode-matched NRD-Sample practices. Covers REBLUR, RELAX, signals, resources, motion, PSR, settings, and probabilistic, checkerboard, or 1 + 1 sampling.
---

# NRD integration review

Review the installed NRD version and enabled rendering modes against primary sources. PSR means primary surface replacement. Unless the user chooses another baseline, select the NRD-Sample branch by primary/PSR sampling mode:

| Sampling at the primary/PSR hit | Sample branch | Review focus |
|---|---|---|
| Probabilistic diffuse/specular lobe selection | `simplex` | Probability bounds, spatial coverage, selection compensation, skipped-lobe distances, reconstruction and prepasses |
| Checkerboard diffuse/specular tracing | `main` | Complementary parity, packed noisy inputs, full logical guide dimensions, frame association |
| `1 + 1`: one diffuse ray and one specular ray per surface sample | `main` | Per-lobe sample budgets, deterministic primary lobe assignment, independent RGB and distances |

Inspect each active mode separately. Probabilistic later bounces do not change deterministic primary/PSR sampling. Use the matching branch for mode-specific estimators and settings.

Use the [checklist](#detailed-nrd-integration-checklist) and [source map](#nrd-review-source-map). Checklist items are questions, not assumed defects.

## Inspect the integration configuration

Locate the installed NRD source or SDK and the consuming application, even if they live in separate repositories. Record:

- Application commit and relevant local modifications; NRD version, commit, and library/shader build configuration.
- Selected denoiser variants, backend, normal/roughness encoding, compile-time feature switches, resource and viewport dimensions.
- Sampling mode, paths per pixel, active PSR/denoising layers, radiance caching, direct/indirect signal composition, and upscaler.

Read the local `README.md`, `Include/NRD.h`, `Include/NRDDescs.h`, `Include/NRDSettings.h`, shader boundary helpers in `Shaders/NRD.hlsli`, and reference `Integration/NRDIntegration.h/.hpp`. Check the latest upstream README and the applicable sample branches through available repository or web tools. Record the revision of each source actually inspected. Follow relevant sample functions rather than copying their settings blindly.

The installed SDK defines the API contract. Treat newer guidance as guidance, not proof that newer APIs are available. Resolve documentation conflicts against the installed implementation.

## Trace the signal and resource paths

Map tracing and accumulation through signal storage, input preparation, NRD dispatch, output unpacking, composition, and AA/upscaling. Follow each active signal and layer, including shared-resource consumers.

Locate the wrapper, settings, shaders, guide producers, signal buffers, composition, and allocator by data flow. Re-locate symbols; old line numbers are not evidence.

Check contracts and data consistency before tuning: normal/radiance encoding, material factors, independent lobe RGB, per-layer hit distances and masks, probability coverage, reconstruction/prepasses, roughness normalization, resource dimensions, and motion. Then assess temporal response and optional quality features.

## Apply the NRD helper baseline

Use the NRD helpers applicable to each signal variant at its representation boundaries. The detailed checklist covers normal/roughness packing, signal packing and unpacking, REBLUR distance normalization, standard reflection material factors, and multi-path specular distance reduction.

A bypass of an applicable NRD boundary helper is a **FAIL** under this skill's helper policy, even if custom arithmetic is equivalent. Label it as policy unless source also proves an API or arithmetic error. Document extensions for BSDFs the standard reflection helper does not represent.

Distinguish the representation consumed by NRD from internal engine buffers and guides consumed by other pipeline stages. Evaluate conversions at the actual NRD boundary and trace shared-resource consumers before recommending changes.

## Evaluate and substantiate

For each applicable check, record status, scope, source locations, and basis: contract, data consistency, helper policy, or optional recommendation. Use:

- **FAIL**: a violated contract, inconsistent data, or missing required helper behavior. State the violated requirement and its effect on the integration.
- **GAP**: an absent optional quality recommendation, such as external confidence, without claiming it is mandatory API input.
- **PASS** (internal evaluation only): the inspected implementation satisfies the relevant invariant. Do not include it in the report.

Keep passes, unused features, and unverified checks out of the report. Explain evidence limits within affected FAIL/GAP findings. Missing runtime evidence alone is not a finding; build success does not establish image quality.

Use statuses without severity scores. A broken usage contract is FAIL. For each finding, explain the affected mode, mechanism, consequence, correction, and verification scenario. Distinguish proven defects from conditional or observed effects; claim reproduction only with evidence.

Compare analogous REBLUR and RELAX settings: sampling contract, reconstruction, prepasses, histories in seconds, confidence, materials, roughness, disocclusion, and input/output units. Do not require equal numeric values for algorithm-specific blur, anti-lag, or history targets. Use installed NRD defaults and the sample branch for the active mode as diagnostic baselines after inputs satisfy their contracts.

## Deliver the review

Write for integration developers. Use navigable HTML unless the user requests another format; honor the requested destination.

Report only FAIL and GAP findings, including in summaries. Omit passes, unused features, pending checks, and the evaluated checklist. Keep revision context relevant and give every evidence cell a short description.

Group findings by contract or component. Put dependencies, evidence limits, and verification steps with the relevant finding. Mention correct behavior only when it explains a failure or gap.
Order corrections by pipeline dependencies, from data and dispatch fixes to quality tuning, without importance scores.

Use navigable finding references and source links where the output format supports them. Link to the repository, revision, and source location actually inspected. Validate references and follow the workspace's formatting conventions.

Default to review/report changes only. Apply renderer fixes or dependency updates only when requested. Identify the inspected sources and qualify each finding with the relevant build/runtime evidence and unverified visual consequences.

## Detailed NRD integration checklist

Apply only relevant checks to the installed version and active modes. Conditional sections do not authorize enabling new features.

Each section names its source basis. Distinguish requirements, sample practices, and optional recommendations.

### Version and configuration

Source basis: NRD README build/API sections; public version macros; `LibraryDesc`; generated `NRDConfig.hlsli`.

- Record the application revision and local changes that affect the reviewed paths.
- Record installed NRD major/minor/build, submodule revision, and actual linked binary version.
- Confirm public headers, library, compiled shader permutations, and dynamic shader sources belong to compatible revisions.
- Read matching local README/header comments and compare newer upstream recommendations without assuming newer APIs exist locally.
- Record each inspected sample branch and revision: `simplex` for primary/PSR probabilistic selection, `main` for checkerboard and `1 + 1`. Keep their mode-specific behavior distinct.
- Enumerate denoiser variants and identifiers, backend(s), active layer count, spp, cache, and upscaler modes.
- Match library normal/roughness encodings with shader configuration and actual texture formats.
- Inspect viewport-offset, checkerboard, confidence, disocclusion-mix, anti-firefly, performance, and quad-intrinsic feature switches where applicable.
- Check backend shader capabilities and generated SPIR-V binding offsets against the actual shader compilation path; custom compilation may legitimately use its own consistent mapping.
- Convert world-unit-dependent defaults when engine units differ from meters; do not scale dimensionless or pixel-valued settings as world distances.

### Resource allocation and lifetime

Source basis: `NRDDescs.h` (`TextureDesc`, pools, `ResourceType`, per-`Denoiser` requirements); README integration/API; official `NRDIntegration.hpp` pool allocation.

- Bind required inputs/outputs for each selected denoiser using the installed per-variant declarations.
- Derive pool formats and dimensions from NRD descriptors instead of stale hardcoded layouts.
- Round both width and height up when dividing resource dimensions by `TextureDesc::downsampleFactor`.
- Test dimensions that are not divisible by the requested downsample factors, including odd sizes, to expose missing edge rows or columns.
- Match input/output resource dimensions and active viewport to the corresponding current/previous settings and per-resource rules; preserve documented exceptions such as lower-resolution confidence textures with linear upscaling.
- Create the SRV/UAV access, sample count, and 2D/mip representation required by NRD; do not infer old mip requirements from another release.
- Preserve permanent-pool history exclusively for its NRD instance until destruction or an intentional reset.
- Reuse/alias transient memory only after the relevant denoising work has completed or is correctly ordered.
- Prevent accidental history sharing between independent denoising layers/instances.
- Reuse shared input textures only after the preceding layer's prepare/denoise/consume sequence is safely ordered.
- Recreate fixed-size allocations when resource size changes, or verify heap, suballocation, or dynamic-resolution-scaling support.
- Recreate or invalidate cached resource views/bindings when their underlying allocations change.
- Gate reads of output pixels outside the viewport/denoising range; NRD may leave them unwritten.
- Preserve OUT_VALIDATION contents between frames for in-place accumulated mini visualizations; inspect reuse across layers.
- Handle allocation/initialization failure before requesting dispatches or consuming invalid outputs; release any partially created instance and resources.

### Descriptors, dispatches, synchronization, and API use

Source basis: `NRD.h` API comments; `PipelineDesc`, `ResourceRangeDesc`, `DispatchDesc`, `InstanceDesc`; official integration implementation.

- Map the installed `ResourceType` enumeration and each dispatch's resources explicitly, without relying on an old slot order.
- Respect descriptor type, resource ranges, range counts, base registers, register spaces, and pipeline layout for each permutation.
- Provide the requested samplers, clamp modes, and filters.
- Load the permutation described by `shaderIdentifier` and the instance-provided entry point; inspect all requested macros.
- Upload each dispatch's constant data with a lifetime that survives queued GPU execution; volatile/ring buffers need sufficient versions.
- Execute the supplied grid dimensions and dispatch order rather than recomputing them from assumed group sizes.
- Consume/copy returned dispatch memory before another `GetComputeDispatches` call overwrites that instance-owned memory.
- Check relevant NRD result values and initialization availability in the application integration.
- Apply common and denoiser-specific settings to the correct instance/identifier before collecting dispatches.
- If using the shipped integration class, follow its NewFrame/settings/denoise lifecycle and its SetCommonSettings/SetDenoiserSettings methods rather than bypassing them with raw NRD API calls. Custom wrappers need equivalent correctness, not identical entry points.
- Serialize calls on a shared shipped integration object according to its thread-safety contract; separately verify any concurrency supported by a custom wrapper.
- Preserve ResourceSnapshot ownership and self-references as documented; propagate the wrapper's final resource states to the application's state tracker or use the supported restoration mechanism.
- Supply texture transitions and UAV dependencies directly or verify that the RHI's binding/state tracking supplies them.
- Order asynchronous work and shared resources across queues; verify that history and noisy inputs are not read before their writers complete.
- Restore or track descriptor heaps, root signatures/layouts, and resource states after native integration changes them.
- Distinguish native integration state handling from RHI-managed synchronization; report an actual missing edge rather than assuming no explicit barrier means no barrier exists.

### Non-noisy guides, matrices, and motion

Source basis: README non-noisy inputs/validation; `CommonSettings`; sample `GetMotion` and primary/PSR guide output in the branch appropriate to the active mode.

- Supply finite non-noisy guides, including pixels not traced this frame and resources reused across layers.
- Use linear view-space Z consistent with the matrices passed to NRD, not device depth or camera-ray distance.
- Use a large finite sky/out-of-range Z rather than INF; confirm skip/composition logic agrees with denoisingRange.
- Make positive/negative view-Z conventions and viewZScale consistent across matrices, guides, and distance normalization.
- Supply world-space normals and linear material roughness before helper encoding.
- Match NRD guide roughness to the primary/PSR tracing BSDF. Follow material remapping, GGX alpha conversion, minimum widths, and delta classification; compare effective linear roughness, not UI or squared values.
- Compare tracing, pre-pack, stored, and decoded roughness under the selected encoding. Allow its expected quantization or helper minimum; encoded zero may not decode to literal zero.
- Verify matrix memory layout, row/column-vector conventions, and any transpositions as a coherent transformation.
- Pass non-jittered current/previous projection matrices and correctly matched camera jitter separately.
- Check jitter units, signs, amplitude, and current/previous frame association against header comments and the actual camera convention.
- Verify `previous = current + motion` after motionVectorScale and optional motionVectorBias.
- For screen motion, scale XY to NRD's normalized viewport coordinates; verify viewport changes and axis orientation.
- For 2.5D motion, provide previous-viewZ minus current-viewZ in Z and a nonzero Z scale.
- For deliberately 2D motion, set Z scale zero and document history rejection risks on moving objects.
- For world-space motion, exclude camera motion; static objects should have zero world motion while camera matrices handle camera movement.
- Remove jitter from the resulting motion exactly once; avoid double compensation through both input generation and bias.
- Check moving-origin/camera-relative world coordinates against previous matrices and scene motion.
- Verify any previous-world transform against the application's coordinate-system changes and the documented feature support.
- Inspect animated/skinned object motion separately from camera-only motion and static-scene reprojection.
- Check roughness floors and disocclusion/upscaler heuristics before NRD packing. Compare the resulting guide with the traced BSDF; matching normalization to an altered guide is insufficient. Evaluate documented physical roughness adjustments separately.

### Boundary helpers and material factors

Source basis: README quick start/material demodulation; `NRD.hlsli` boundary helpers; sample material-factor generation and Composition. Helper-compliance status follows the policy above and applies only to relevant signal variants.

- Encode the NRD normal/roughness texture with `NRD_FrontEnd_PackNormalAndRoughness`.
- Decode every active app-side consumer of that texture with `NRD_FrontEnd_UnpackNormalAndRoughness`.
- Confirm encoding supports the selected texture format and material-ID representation; do not treat packed `.xyz`/`.w` as raw normal/roughness.
- Distinguish the NRD guide from internal normal encodings and other denoiser guides before flagging custom packing.
- Pack REBLUR and RELAX noisy signals using the matching front-end functions and appropriate alpha units.
- Unpack each output using the matching back-end helper before remodulation/composition.
- Use `NRD_MaterialFactors` for standard reflection radiance/SH factors; flag custom-only factors as a helper-policy FAIL. Do not require radiance demodulation for scalar occlusion, SIGMA penumbra, or generic reference signals.
- Supply actual material albedo, F0, normal, view direction, and material roughness to that helper, not previously integrated estimates masquerading as its inputs.
- Use identical factors for demodulation and remodulation, including clamps and minimum shaping.
- Keep material-factor roughness consistent before/after filtering; NRD's filtering-adjusted roughness need not be its material roughness.
- Check dark albedo, grazing angles, low roughness, metals, and near-zero factors for finite arithmetic and intended bias.
- Recognize that NRD's factor shaping differs from a simple max clamp; evaluate any extra clamp without silently changing one side of the round trip.
- Document reflection/transmission/custom-BSDF extensions rather than assuming the standard reflection helper represents every material model.
- Remove PSR throughput from the noisy signal as far as practical and apply symmetric remodulation; check near-zero throughput.
- Preserve intended behavior for other consumers of shared material or guide storage when introducing NRD-specific factors.
- Verify optional sanitization is configured deliberately and finite guide construction does not rely on noisy-signal sanitization to repair unrelated data.

### Radiance construction and lobe separation

Source basis: README noisy inputs/material demodulation/combined lighting; independent diffuse/specular results and estimators in the applicable sample branch.

- Separate radiance by the diffuse/specular lobe at the primary or PSR surface, not by an unrelated later bounce.
- Keep diffuse and specular RGB independent through storage and accumulation; identify lossy decomposition and lobe contamination separately from total-energy changes.
- Trace NEE, BSDF-sampled emission, cache returns, ReSTIR results, and multiple-spp accumulation into the correct lobe.
- Decouple primary-surface emission and stable sky paths where appropriate so NRD does not blur sharp emissive detail.
- Normalize radiance for the actual estimator: probabilistic primary/PSR selection includes selection compensation and averages over its full path budget, including skipped-lobe zero contributions; separate diffuse/specular tracing averages each lobe over its own sample budget. Do not halve `1 + 1` signals by dividing each by the doubled total ray count.
- Keep primary/PSR lobe classification stable through later bounces and path restarts.
- Check nonnegative, finite linear radiance within the active viewport and denoising range before packing. Allow black/skipped-lobe zeros and respect FP16/second-moment limits.
- Distinguish signed encoded coefficients and noisy NaN/Inf outside processed regions from non-noisy guides, which must remain finite.
- Check the exposure applied at the NRD boundary against the documented input convention and temporal-history behavior.
- Check clamp/compression behavior under brightness and exposure changes, and its symmetry where an inverse transform is required.
- Treat sparse high-energy probability-weighted fireflies as a sampling problem before relying on anti-firefly settings.
- Verify optional suppression/attenuation factors are intentional and applied to the correct layer without assuming every shared-buffer contribution is duplicated.

### Hit distances and multiple-path reduction

Source basis: README noisy inputs/combined lighting and shader pseudocode; helper contracts; selected-lobe distance construction and reduction in the applicable sample branch.

- Verify sampled diffuse paths supply representative in-lobe distances and that reconstruction has valid source samples where required.
- Verify specular hit-distance guidance covers every denoised signal, layer, and supported roughness range.
- Exclude the camera-to-primary/PSR segment from indirect hit distance.
- Use the first segment after primary/PSR as the baseline before introducing more complex multi-segment accumulation.
- Associate each distance with a ray in the actual denoised BRDF lobe.
- Preserve zero for a skipped lobe through storage, normalization, packing, and any app-side filtering.
- Check contact/inside-surface zero handling separately from missing samples; avoid replacing zeros with epsilon or a constant distance. If optional `NRD_FrontEnd_TrimHitDistance` is used, evaluate its threshold, units, and representative-distance bias; absence alone is not a failure.
- Reject negative/nonfinite distances and inspect miss/sky sentinel conversion through the matching helper.
- Never apply radiance-style selection-probability, PDF, BRDF, or throughput compensation to hit distance. Use the signal's prescribed per-lobe reduction rather than dividing by the total number of unrelated paths.
- For advanced MIS/RIS/ReSTIR samples, verify BRDF-aware probabilistic acceptance and an in-lobe fallback instead of indiscriminately using selected-light distance.
- Preserve ReSTIR disocclusion zeros as missing samples for the supported reconstruction path.
- For combined direct/indirect specular signals, inspect indirect in-lobe distance as the tracking baseline; an NEE light distance can describe the wrong lobe.
- For combined diffuse lighting, identify direct-light producers (such as NEE or reservoir sampling) and whether each has distance guidance at the same primary/PSR surface and layer. An indirect-only guide can blur direct shadows when indirect rays miss or reach distant geometry.
- If direct-light distances guide diffuse filtering, check visible local lights, shadowed samples, and directional/infinite lights. When using blocker distance in shadow, test whether the visibility query returns representative geometry; accept-first traversal may return an arbitrary hit. Account for ray-origin offsets and light endpoints.
- For a direct/indirect diffuse mix, compare weights on the same linear estimator scale and keep the result finite. Check lobe separation, light-selection/MIS weights, throughput, and attenuation. Test whether potential pre-visibility weighting retains shadowed blocker guidance better than visible-only luminance; the README does not mandate either formula. Do not invent a direct distance when none is valid.
- With multiple sampled lights, check whether the distance reduction represents the lights that drive the diffuse signal. An unweighted average can let a distant low-contribution light dilute a nearby dominant blocker; test bright near and dim far lights in both visible and shadowed pixels.
- Treat the README's 0.5 direct-mix cap as an adjustable example, not a required limit. Keep the mix finite and in [0, 1], and test whether any chosen cap preserves contact shadows when an indirect ray misses and its REBLUR distance normalizes to 1. Do not assume specular and diffuse need the same mixing policy.
- Adjust any accumulated later segments for curvature and lobe spread/energy dissipation; do not accept raw sum-of-all-lengths by default.
- Verify application-side filtering or reconstruction preserves sampled-lobe association and satisfies the documented missing-sample contract.
- Normalize REBLUR distances with `REBLUR_FrontEnd_GetNormHitDist`; supply RELAX distances in scene units.
- Match shader hit-distance normalization parameters to the denoiser settings and engine world units.
- Match specular normalization roughness to the effective NRD guide roughness, allowing expected encoding quantization; matching other parameters does not compensate for a roughness mismatch.
- Use diffuse roughness 1 for diffuse normalization under the inspected helper contract.
- Average diffuse hit distances over diffuse paths, and specular occlusion distances over specular paths where prescribed; preserve zero when that lobe has no samples and do not count other-lobe paths.
- For multiple specular radiance/SH samples of the same pixel/layer, use `NRD_FrontEnd_SpecHitDistAveraging_*` and inspect its zero/minimum semantics. Count eligible specular samples, not total rays; base `1 + 1` has only one specular sample.
- For REBLUR, follow normalization before signal-specific reduction, or establish equivalent ordering with a consistent scale, saturation, and zero handling; ordering alone does not prove a defect.
- For combined diffuse guidance, normalize each eligible direct and indirect distance with the same layer view depth, hit-distance parameters, and diffuse roughness before REBLUR reduction or mixing; preserve RELAX distances in scene units. Verify any alternative ordering against saturation and zero handling rather than assuming normalization commutes with averaging.
- Reduce specular radiance distances only over relevant specular paths; do not dilute them by counting diffuse/skipped paths or substitute the occlusion averaging rule.
- Trace distance initialization and updates across paths to ensure per-path setup cannot overwrite accumulated representative distances.
- For specular radiance/SH minimum-distance reduction, verify order independence using two eligible paths with different distances; ensure later farther samples cannot overwrite the representative nearer distance. Verify diffuse and specular-occlusion averages against their own reduction rules.

### Probabilistic lobe selection and low-discrepancy sampling

Source basis: `HitDistanceReconstructionMode`, per-variant prepass availability, README noisy inputs/blue noise/resolution guidance, `simplex` Bayer/Weyl selection. Primary/PSR probability-coverage and reconstruction requirements below apply to probabilistic selection, not checkerboard or `1 + 1` sampling.

- Identify whether selection is probabilistic at primary/PSR, later bounces only, checkerboarded, or fully sampled per lobe.
- For probabilistic primary/PSR selection, enable supported hit-distance reconstruction.
- Enable nonzero diffuse and specular prepass radii for that sampling scheme where the selected variant exposes those passes; the presence of fields in a shared settings structure does not establish pass availability.
- Tune prepass radii for sparse-lobe coverage and contact detail. Large diffuse radii can soften shadows; the specular prepass also estimates reflection-tracking distance. Check README/header conditions before disabling diffuse blur or using specular-motion-only mode. Sample radii are not universal minima.
- For lobe probabilities between zero and one, enforce the installed reconstruction footprint's documented bounds.
- Preserve deterministic zero/one cases, such as absent diffuse on metals, without dividing by zero.
- At metal/dielectric boundaries, test whether hit-distance reconstruction draws diffuse samples across the boundary; the reconstruction pass may not compare material IDs. With a material-ID-capable encoding, pack meaningful IDs and configure `minMaterialForDiffuse` (and `minMaterialForSpecular` where needed) to limit later cross-material mixing. Report any reconstruction leakage separately.
- Supply a spatial coverage guarantee such as the sample's Bayer policy; independent per-pixel RNG/temporal Sobol sampling alone is not that guarantee.
- Add suitable screen-uniform temporal jitter/decorrelation to prevent persistent lobe-selection/low-discrepancy directional bias.
- Recompute throughput/PDF compensation consistently if probabilities or selection strategy change.
- Check coverage on each active PSR layer and among geometrically eligible neighbors, not merely in a full-screen binary mask.
- Bound excessive probability-weighted energy for later bounces according to the README guidance while preserving deterministic cases.
- Evaluate low-discrepancy sequence length against the denoiser history window and reference convergence.
- Preserve spatial blue-noise properties when deriving path/bounce/sample sequences; inspect per-pixel scrambles and screen-uniform rotations deliberately.
- Avoid interpreting blue-noise temporal sampling as proof that every sparse lobe has valid neighboring distance data.
- Check separately whether subsampling/adaptive tracing leaves guides valid and whether missing radiance has the compensation/reconstruction policy NRD expects.

### Checkerboard and 1 + 1 sampling

Source basis: `CheckerboardMode` and reconstruction comments in `NRDSettings.h`; `main` sample TraceOpaque/WriteResult and mode-dependent NRDSample settings.

- Distinguish checkerboard sampling from full-resolution `1 + 1` and probabilistic primary/PSR selection; identify the actual diffuse and specular sample budgets per pixel/layer.
- Keep checkerboard noisy inputs and ordinary guides at the documented logical full resolution, preserving per-resource exceptions such as lower-resolution confidence. Pack noisy lobe signals horizontally into the left half of their textures; keep guides on the full pixel grid and let NRD perform its checkerboard upsampling.
- Match `BLACK`/`WHITE` orientation, even-frame convention, pixel parity, and `CommonSettings::frameIndex` between tracing and NRD. Combined diffuse/specular variants use complementary orientations.
- Use only valid packed samples for each lobe; do not treat untraced checkerboard cells as full-resolution zero samples or halve logical `resourceSize`/`rectSize`.
- Keep `hitDistanceReconstructionMode` OFF for checkerboard as required by the installed contract. For fully sampled `1 + 1`, keep `checkerboardMode` and `hitDistanceReconstructionMode` OFF; do not apply probability-footprint bounds or Bayer requirements to deterministic primary lobe assignment.
- Do not apply inverse primary lobe-selection compensation to checkerboard parity or deterministic `1 + 1` assignment. Preserve BRDF/PDF weighting and any later-bounce probabilistic compensation required by the actual estimator.
- Normalize each `1 + 1` lobe by its own sample budget, preserving independent RGB and hit distances. Do not trigger multiple-specular-path reduction merely because two total rays are traced.
- Evaluate prepasses using the selected variant's contract and input quality; do not import the probabilistic mode's mandatory nonzero-radii rule into deterministic modes.
- Inspect checkerboard RNG advancement and temporal sequence reuse for each lobe's valid samples while NRD `frameIndex` advances per rendered frame; do not prescribe a sample-specific RNG formula universally.
- When switching sampling modes, update packing, estimator normalization, and denoiser settings together; reset history or rebind/recreate resources when the changed layout or history interpretation requires it.

### Primary surface replacement and layered denoising

Source basis: README PSR section; `NRD.hlsli` PSR throughput comments; virtual geometry and Composition in the applicable sample branch.

- Identify the primary or PSR surface represented by each layer and the application's replacement/termination criterion; check the guide, material, motion, and distance origin coherently rather than inferring replacement solely from numeric roughness.
- Replace associated material properties, normal/roughness, viewZ, motion, distance origin, and directions coherently for PSR.
- Use the replacement surface's effective tracing roughness for its PSR guide. A pure mirror reflecting a rough surface does not imply zero roughness at the replacement surface; do not substitute the original mirror's roughness for that material.
- Place virtual geometry along the camera view ray using the intended optical-distance model.
- Transform terminal normals through the reflection chain in the proper order and preserve reflection handedness.
- Transform object motion into the corresponding virtual frame and distinguish it from camera motion.
- Test planar mirrors before diagnosing the harder curved/refraction cases.
- Inspect curvature/spread adjustment before and after PSR; characterize omitted curvature as a conditional quality limitation where the README permits it.
- Do not assume a primary surface's curvature is still available to NRD after replacing its guides.
- Use alternate disocclusion thresholds only where PSR needs them; avoid relaxing ordinary surfaces.
- Evaluate motion and roughness heuristics designed for another denoiser or upscaler before reusing their guides for NRD.
- Give each noisy layer its own corresponding distance and history, including non-dominant reflected/transmitted layers.
- Verify composition includes remodulation/throughput once and preserves stable emission/sky contributions.
- Check branch/layer changes, newly appearing layers, and unavailable history without inventing a need for complex extra state tracking absent from NRD's contract.

### REBLUR settings and output interpretation

Source basis: `ReblurSettings`, `ReblurHitDistanceParameters`, README REBLUR recommendations, sample settings for the active sampling mode.

- Verify normalized-distance units, normalization-parameter domains, and effective roughness consistency throughout encoding and decoding.
- Verify reconstruction and prepass settings support the actual sampling mode.
- Check main, fast, stabilization, and history-fix frame ranges and relationships against current headers.
- Evaluate stabilization and responsive-accumulation behavior deliberately; do not equate all history counters.
- Inspect convergence, blur radius, lobe-angle, roughness, and plane-distance settings only after input correctness is established.
- Evaluate anti-lag and anti-firefly settings against input noise and external confidence rather than using them to hide metadata errors.
- If using prepass only for specular motion estimation, verify it is appropriate for the signal and sampling mode rather than a universal quality improvement.
- Configure material comparisons and special material IDs for the classes actually supplied by the application.
- Interpret output alpha according to whether it represents normalized occlusion or history length; do not assume an always-valid distance or color value.
- Compare performance mode as a deliberate quality/performance tradeoff and verify matching shader/library switches.

### RELAX settings and comparisons with REBLUR

Source basis: `RelaxSettings`, README RELAX recommendations, sample settings and temporal setup for the active sampling mode.

- Supply scene-unit distances rather than REBLUR-normalized alpha.
- Verify reconstruction and both prepasses remain enabled for primary/PSR probabilistic sampling where the selected variant exposes them; check deterministic modes against their own requirements.
- Compare diffuse/specular main and fast histories using intended accumulation durations, not just frame counts.
- Enforce the installed history-fix domain and its relation to fast history; inspect slider/load paths, not just defaults.
- Evaluate A-trous iteration count, lobe/roughness rejection, depth threshold, and luminance weights against the actual noisy signal.
- Distinguish algorithm-specific thresholds from analogous REBLUR settings; numerical equality is not a contract.
- Tune custom anti-lag acceleration, spatial sigma, temporal sigma, and reset amounts after input/temporal baselines are established.
- Check whether confidence-driven controls have any active confidence input to operate on.
- Compare anti-firefly quality/cost and finite second-moment input range.
- Inspect roughness edge stopping, material comparison thresholds, and low-confidence relaxation using the classes and guides actually bound.

### Temporal history, confidence, and reset behavior

Source basis: `CommonSettings`, `GetMaxAccumulatedFrameNum`, history-setting comments; README confidence/frame-generation/other recommendations; sample gradients/blur/FPS setup.

- Increment frameIndex once per rendered frame, preserving its checkerboard association; do not increment per layer/dispatch.
- Pass current/previous guides, jitter, matrices, and dimensions from the intended frames.
- Express target accumulation in seconds and use smoothed rendered FPS when converting to frame counts.
- Exclude generated frames from the FPS used for NRD accumulation.
- Clamp converted histories to the selected denoiser's limits and preserve supported main/fast/history-fix modes.
- For `timeDeltaBetweenFrames`, pass the immediate, unfiltered previous rendered-frame time in milliseconds when positive; use a nonpositive value for NRD internal timing. Smooth rendered FPS only when converting history durations to frame counts.
- Use RESTART for an ordinary initialized-history reset and CLEAR_AND_RESTART or proper clears when history may contain garbage.
- Verify resets occur for the intended frame and do not accidentally persist or restart every frame after UI/mode events.
- Treat external confidence as an optional but strongly recommended quality input, not a mandatory API resource when its flag is false.
- If available, compute lighting gradients from previous-frame stored and retraced lighting using matching previous RNG seeds.
- Verify gradient computation isolates lighting changes from sampling differences and appropriately relaxes dynamic-object mismatches.
- Inspect geometry/normal-aware spatial filtering and confidence conversion, avoiding unnecessary temporal lag in the gradient itself.
- Bound confidence to 0..1 and make flags, compile-time support, bindings, resolution, and resource contents agree.
- Allow lower-resolution confidence with the documented linear upscaling; evaluate shared lighting confidence for probabilistic one-path sampling and separate diffuse/specular confidence for `1 + 1` without requiring either policy universally.
- Calibrate confidence mapping separately for REBLUR and RELAX; the same raw gradient need not use the same final value.
- Do not mistake disocclusion-threshold mix, a radiance cache, or final-upscaler history controls for NRD confidence inputs.
- Check lighting-step response and convergence in seconds at multiple rendering FPS, alongside validation history-length views.

### Dynamic resolution and upscaling

Source basis: `CommonSettings` comments; README resolution/upscaling guidance; sample guide/compose/upscaler passes for the active mode. Checkerboard packing is covered separately above.

- Separate tracing density, NRD processing resolution, and output/upscaler resolution in the review.
- Match resourceSize/rectSize current and previous values to actual allocations and active viewports.
- For viewport offsets, inspect installed input/output origin semantics, exclusions, and compile-time enablement.
- Keep both origin-plus-rect extents inside resource bounds and apply required reset when output origin/history addressing changes.
- Recognize that denoising resolution is configured per instance; signals denoised at different sizes need appropriate separate instances.
- Supply downsampled guides at NRD resolution while retaining full-resolution guides for application-side resolve when needed.
- Evaluate SH/SG resolve and re-jittering for detail recovery at reduced processing resolution; supported non-SH variants remain valid.
- Verify SH/SG output resources, matching unpacking, direction transforms, resolve, and material modulation when the variant is enabled.
- Distinguish directional resolve from texture resizing; application-side spatial upsampling still needs a correct implementation.
- Decode packed NRD normals through the helper for resolve/re-jitter neighbor reads and handle boundaries/geometry appropriately.
- Verify application AA/upscaling jitter, motion, exposure, and resolution conventions agree with the composed denoised image.

### SIGMA

Source basis: SIGMA resource declarations in `NRDDescs.h`, `SigmaSettings`, penumbra/translucency helpers, README SIGMA guidance, and the sample's `USE_BLUE_NOISE_FOR_SHADOWS` / `GetBlueNoise` path. Apply these checks when SIGMA is used.

- As a quality check, require spatial blue-noise low-discrepancy sampling for SIGMA shadows, using a short animated period or a static screen-space pattern. The sample uses four frames; the README recommends 4-8 animated frames or static. Assess shimmer before reusing a long radiance sequence. Absence is a GAP under this review policy, not an API-contract FAIL.
- Construct penumbra/translucency with SIGMA front-end helpers and unpack its matching outputs.
- Cast shadow visibility rays from the receiver and implement the documented hit/miss/NoL conventions.
- Avoid ACCEPT_FIRST_HIT_AND_END_SEARCH where it would produce an arbitrary occluder distance instead of the required representative hit.
- Match directional/local light geometry, angular size, and units to the chosen penumbra helper.
- Check SIGMA stabilization/output-history lifetime, motion requirements, and checkerboard support against the installed release.
- If combining multiple lights, document the supported approximation and limitations instead of assuming generic unbiased multilighting shadow denoising.

### Occlusion-only denoising

Source basis: occlusion `Denoiser` and `ResourceType` declarations in `NRDDescs.h`; `NRD.hlsli` normalized-distance and directional-occlusion helpers; README shader pseudocode and AO/SO sampling guidance; mode-appropriate sample occlusion paths. Inspect the selected variant's dispatches when shared settings leave pass availability unclear.

- Identify scalar diffuse, scalar specular, combined scalar, or directional diffuse occlusion from the installed denoiser declarations. Do not assume that a radiance variant or another denoiser family supplies an equivalent occlusion mode.
- Bind the selected scalar IN_DIFF_HITDIST/IN_SPEC_HITDIST and matching outputs, or the directional IN_DIFF_DIRECTION_HITDIST/OUT_DIFF_DIRECTION_HITDIST pair, with the required ordinary guides and enabled optional resources.
- Construct diffuse and specular occlusion from the documented cosine-weighted and VNDF lobe sampling, or an explicitly justified importance-sampling equivalent. Match the sampled lobe to the supplied normal and roughness.
- Encode normalized hit distances with REBLUR_FrontEnd_GetNormHitDist; match the denoiser's normalization parameters and effective guide roughness, using diffuse roughness 1. Keep the scalar signal distinct from radiance, scene-unit hit distance, and an arbitrary visibility mask.
- Trace hit, miss, inside-surface, and skipped-lobe cases through normalization and packing. Exclude the primary/PSR segment and preserve the documented meaning of zero; verify any miss-sentinel conversion and finite active inputs.
- Reduce scalar diffuse/specular occlusion distances over the corresponding lobe's samples as prescribed by the README/sample. Do not apply radiance probability/PDF compensation or replace specular occlusion averaging with the specular radiance minimum-distance helpers.
- For directional occlusion, reduce directions and distances according to the selected estimator, then encode them with REBLUR_FrontEnd_PackDirectionalOcclusion. Preserve the helper's occlusion-weighted directional coefficients in storage; do not reinterpret packed coefficients as a normalized bent normal.
- Use formats that preserve each signal's range, including signed directional coefficients. Do not let an unsigned normalized format clamp negative directional components.
- Decode directional output with REBLUR_BackEnd_UnpackDirectionalOcclusion and use the intended directional resolve. Treat scalar output as normalized occlusion; do not apply radiance decoding or material demodulation/remodulation to it.
- Verify pass availability per installed variant. Scalar occlusion uses a reduced pipeline without radiance prepasses or temporal stabilization; directional diffuse occlusion can expose those passes. Do not assume a shared ReblurSettings field affects every occlusion variant.
- Apply reconstruction and skipped-lobe rules only for supported probabilistic sampling; use the main-branch checkerboard or 1 + 1 contracts for deterministic modes. Do not require a nonzero prepass radius when the selected scalar variant has no prepass.
- Evaluate spatial blue-noise LDS for occlusion-only AO/SO; it may improve stability and allow shorter history than white noise.
- Tune sequence length with accumulation (the sample uses eight frames for occlusion). Check sample budget, checkerboard parity, contact detail, and motion response before shortening history; SIGMA's four-frame sequence is not an AO/SO requirement.
- Check optional confidence, material comparison, and disocclusion-mix support against the selected variant's declarations. Verify any history/debug output override before interpreting the result as occlusion.
- Verify contact geometry, open/missed rays, roughness ramps for specular occlusion, and per-lobe multiple-sample reduction. For directional output, test known directions with negative components and confirm the intended bent-direction/lighting response after decoding and resolve.

### Hair and transparency

Source basis: README hair guidance and sample special-material and transparency paths. Keep unused features in the internal scope evaluation, outside the problem report.

- For hair/grass, inspect a suitable guide normal derived from strand/view geometry, semantic material IDs, and strand scale rather than passing raw tangents as normals.
- Treat volumetrics/transparency as a separately designed signal path; do not infer native support from the existence of a glass sample.
- Evaluate sample-inspired cached/reprojected transparent rendering only if that mode is within the application's intended scope.

### Exposed settings and cross-field validation

Source basis: all relevant public-header ranges and relationships; README defaults/tuning guidance; application settings/UI/serialization paths.

- Check slider endpoints, free numeric entry, presets, command-line settings, and saved-state loads against header domains.
- Verify that REBLUR hit-distance parameters satisfy their documented ranges, including values supplied through UI controls and configuration files.
- Validate main/fast/history-fix relationships without forbidding explicitly supported disabled modes.
- Keep reconstruction and prepass availability consistent with the active sampling mode after runtime changes.
- Validate disocclusion thresholds and mix range while distinguishing recommended tuning ranges from strict numerical-domain requirements.
- Distinguish shared common settings from method-specific settings and ensure each method receives the correct structure.
- Reinitialize/rebind resources or histories when a mode change actually requires it; do not assume every setting requires destruction.
- Restore supported NRD defaults as a diagnostic baseline before retaining heavily customized settings across library upgrades.
- Place proposed input validation in the appropriate application or validation layer and respect the target project's architectural boundaries.

### Evidence and visual verification

Source basis: README validation guidance and public validation-output comments.

- Use NRD Debug checks and OUT_VALIDATION where available; record which diagnostics were actually enabled.
- Capture raw lobe RGB, sampled masks, alpha units, decoded normals/roughness, viewZ, motion, factors, and settings for relevant failure modes.
- Test contact geometry and independent diffuse/specular colors.
- Capture tracing roughness, pre-pack roughness, and decoded NRD roughness for exact-zero materials, values around the tracing delta threshold, and a roughness ramp. Include PSR/disocclusion boundaries and check the selected encoding's expected precision separately from application-side inflation.
- Test low-F0 dielectrics beside metals to expose lobe coverage and material-boundary mixing.
- Inspect every active PSR layer, including non-dominant reflected/transmitted branches.
- Test camera-only motion, animated objects, planar mirrors, and bounded curved/transmissive limitations separately.
- Test supported single- and multiple-path sample counts with known distance differences and path-order permutations.
- Test lighting changes and convergence across representative rendered frame rates, including frame-generation behavior when supported.
- For mixed direct/indirect diffuse signals, compare raw input, guide distance, and denoised contact shadows with a dominant local or directional light, an occluded sample, and an indirect environment miss. Vary light and indirect intensity separately to expose weighting or prepass blur that masks a correct distance.
- Test standard, odd, and upscaler-selected render dimensions to expose underallocation/viewport errors.
- Test material-factor arithmetic with filtering bypassed before assessing denoised image differences.
- Test brightness/exposure sweeps and finite HDR behavior without assuming a clamp is energy-preserving.
- Compare equivalent captured inputs under REBLUR and RELAX before changing several unrelated tuning knobs.

## NRD review source map

Use the installed NRD sources and mode-matched sample as primary sources. The links below point to NRD's default branch and sample `simplex`/`main`; cite the revisions actually inspected.

### Source hierarchy

1. Installed NRD public-header declarations/comments, build configuration, and boundary-helper implementation determine the application's actual API and encoding contract.
2. The matching local README and current upstream README explain signal contracts, recommended practices, and newer guidance. Keep version differences explicit.
3. NRD-Sample `simplex` shows probabilistic primary/PSR selection; `main` shows checkerboard and `1 + 1`. Match the active mode. A difference from the sample alone does not prove a defect.
4. The shipped NRD integration implementation is the reference for resource allocation, descriptor mapping, and dispatch/lifetime mechanics. Inspect NRD internal passes only to resolve a specific uncertainty or establish a concrete consequence.

Resolve source conflicts against the installed implementation.

### Local files to inspect

| Source | Relevant symbols and comments |
|---|---|
| `README.md` | Inputs; noisy/non-noisy contracts; material demodulation; low-discrepancy sampling; PSR; resolution/upscaling; confidence; combined direct/indirect lighting; other recommendations |
| `Include/NRD.h` | Version macros; API results; settings/dispatch call sequence; returned dispatch-memory lifetime |
| `Include/NRDDescs.h` | `ResourceType`; per-`Denoiser` input/output lists; normal/roughness encodings; `TextureDesc::downsampleFactor`; `PipelineDesc`; `InstanceDesc`; `DispatchDesc`; validation output lifetime |
| `Include/NRDSettings.h` | `CommonSettings`; `CheckerboardMode`; `HitDistanceReconstructionMode`; `ReblurHitDistanceParameters`; history ranges/relationships; confidence and material switches; SIGMA settings when applicable |
| `Shaders/NRD.hlsli` and `NRDConfig.hlsli` | Guide pack/unpack; material factors and minima; distance normalization; zero/sanitization behavior; multiple-path distance reduction; radiance/SH/SG helpers and compile-time encoding |
| `Integration/NRDIntegration.h/.hpp` | Resource snapshots; resource-state handling; native wrapper behavior; pool dimensions rounded up; samplers; descriptors; resizing; per-frame lifecycle |

### NRD

- [README](https://github.com/NVIDIA-RTX/NRD/blob/master/README.md)
- [NRD.h](https://github.com/NVIDIA-RTX/NRD/blob/master/Include/NRD.h)
- [NRDDescs.h](https://github.com/NVIDIA-RTX/NRD/blob/master/Include/NRDDescs.h)
- [NRDSettings.h](https://github.com/NVIDIA-RTX/NRD/blob/master/Include/NRDSettings.h)
- [NRD shader helpers](https://github.com/NVIDIA-RTX/NRD/blob/master/Shaders/NRD.hlsli)
- [Integration allocation and dispatch implementation](https://github.com/NVIDIA-RTX/NRD/blob/master/Integration/NRDIntegration.hpp)

### Simplex navigation: probabilistic primary/PSR selection

| File / search terms | Practice to inspect |
|---|---|
| [TraceOpaque.cs.hlsl](https://github.com/NVIDIA-RTX/NRD-Sample/blob/simplex/Shaders/TraceOpaque.cs.hlsl): `EstimateDiffuseProbability`, `Bayer4x4`, `Weyl1D` | Probability bounds, spatial coverage, temporal decorrelation, and throughput compensation |
| Same file: `accumulatedHitDist`, `ApplyThinLensEquation`, `accumulatedDiffuseLikeMotion` | First-bounce distance baseline and curvature/lobe-spread-aware multi-segment accumulation |
| Same file: `result.diffRadiance`, `result.specRadiance`, `GetNormHitDist` | Independent lobe RGB and distances, correct selected-lobe normalization, skipped-lobe zeros |
| Same file: `Xvirtual`, `GetMotion`, `GetMaterialID`, `PackNormalAndRoughness` | PSR guides, virtual motion, material classification, helper encoding |
| [Shared.hlsli](https://github.com/NVIDIA-RTX/NRD-Sample/blob/simplex/Shaders/Shared.hlsli): `GetMotion` | Previous-minus-current motion convention and view-depth delta |
| [RaytracingShared.hlsli](https://github.com/NVIDIA-RTX/NRD-Sample/blob/simplex/Shaders/RaytracingShared.hlsli): `GetMaterialFactors`, `NRD_MaterialFactors`, blue-noise symbols | Material-factor policy, sampling, and model-specific extensions |
| [Composition.cs.hlsl](https://github.com/NVIDIA-RTX/NRD-Sample/blob/simplex/Shaders/Composition.cs.hlsl) | Matching output unpacking, material/PSR remodulation, SH/SG resolve and re-jittering |
| [NRDSample.cpp](https://github.com/NVIDIA-RTX/NRD-Sample/blob/simplex/Source/NRDSample.cpp): `GetDefaultRelaxSettings`, `GetDefaultReblurSettings` | Consistent sampling/reconstruction policy and semantic material comparisons |
| Same file: `adaptiveAccumulation`, `GetMaxAccumulatedFrameNum`, `SetCommonSettings`, `NewFrame` | Smoothed rendered FPS, frame/settings lifecycle, dimensions, jitter, validation |
| [SharcUpdate.cs.hlsl](https://github.com/NVIDIA-RTX/NRD-Sample/blob/simplex/Shaders/SharcUpdate.cs.hlsl) and [ConfidenceBlur.cs.hlsl](https://github.com/NVIDIA-RTX/NRD-Sample/blob/simplex/Shaders/ConfidenceBlur.cs.hlsl) | Previous-frame lighting gradients and geometry-aware confidence filtering |
| [TraceTransparent.cs.hlsl](https://github.com/NVIDIA-RTX/NRD-Sample/blob/simplex/Shaders/TraceTransparent.cs.hlsl) | Optional transparent rendering approach; do not assume NRD natively supports volumetric/transparency denoising |

### Main navigation: checkerboard and 1 + 1

| File / search terms | Practice to inspect |
|---|---|
| [TraceOpaque.cs.hlsl](https://github.com/NVIDIA-RTX/NRD-Sample/blob/main/Shaders/TraceOpaque.cs.hlsl): `pathNum`, `RESOLUTION_FULL`, `isDiffuse` | Two-path `1 + 1` budget, checkerboard/path parity, deterministic primary assignment versus later-bounce probabilistic selection |
| Same file: `radianceNorm`, `diffNorm`, `SpecHitDistAveraging` | Per-lobe radiance budgets, diffuse distance averaging, specular radiance reduction versus occlusion averaging |
| Same file: `NRD_MODE`, `OCCLUSION`, `DIRECTIONAL_OCCLUSION`, `PackDirectionalOcclusion`, `WriteResult` | Scalar versus directional occlusion paths, per-lobe distance reduction, and matching signal output packing |
| Same file: `WriteResult`, `RESOLUTION_HALF`, `checkerboard` | Left-half packed noisy outputs, complementary diffuse/specular cells, full-resolution guide writes |
| [NRDSample.cpp](https://github.com/NVIDIA-RTX/NRD-Sample/blob/main/Source/NRDSample.cpp): `GetDefaultRelaxSettings`, `GetDefaultReblurSettings`, `tracingMode` | Mode-dependent checkerboard orientation, reconstruction/prepass settings, and settings changes on mode switches |
| Same file: `resourceSize`, `rectSize`, `SetCommonSettings` | Full logical NRD dimensions despite checkerboard noisy-signal packing |
| [Shared.hlsli](https://github.com/NVIDIA-RTX/NRD-Sample/blob/main/Shaders/Shared.hlsli): `RESOLUTION_FULL`, `RESOLUTION_FULL_PROBABILISTIC`, `RESOLUTION_HALF` | Mode identifiers and shared shader configuration |
| [RaytracingShared.hlsli](https://github.com/NVIDIA-RTX/NRD-Sample/blob/main/Shaders/RaytracingShared.hlsli): `GenerateRayAndUpdateThroughput`, `gFrameIndex` | Direction sampling, throughput updates, and checkerboard-aware temporal sequence indexing |
