/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "NRDIntegration.h"

#ifdef _WIN32
#    include <malloc.h>
#else
#    include <alloca.h>
#endif

static_assert(NRD_VERSION_MAJOR >= 4 && NRD_VERSION_MINOR >= 15, "Unsupported NRD version!");
static_assert(NRI_VERSION >= 165, "Unsupported NRI version!");

#define NRD_INTEGRATION_RETURN_FALSE_ON_FAILURE(expr) \
    if ((expr) != nri::Result::SUCCESS) \
    return false

namespace nrd {

constexpr uint32_t SET_ALL = 0;
constexpr uint32_t SET_SAMPLERS = 1;

constexpr uint32_t RANGE_TEXTURES = 0;
constexpr uint32_t RANGE_STORAGES = 1;
constexpr uint32_t RANGE_SAMPLERS = 2;

constexpr std::array<nri::Format, (size_t)Format::MAX_NUM> g_NrdFormatToNri = {
    nri::Format::R8_UNORM,
    nri::Format::R8_SNORM,
    nri::Format::R8_UINT,
    nri::Format::R8_SINT,
    nri::Format::RG8_UNORM,
    nri::Format::RG8_SNORM,
    nri::Format::RG8_UINT,
    nri::Format::RG8_SINT,
    nri::Format::RGBA8_UNORM,
    nri::Format::RGBA8_SNORM,
    nri::Format::RGBA8_UINT,
    nri::Format::RGBA8_SINT,
    nri::Format::RGBA8_SRGB,
    nri::Format::R16_UNORM,
    nri::Format::R16_SNORM,
    nri::Format::R16_UINT,
    nri::Format::R16_SINT,
    nri::Format::R16_SFLOAT,
    nri::Format::RG16_UNORM,
    nri::Format::RG16_SNORM,
    nri::Format::RG16_UINT,
    nri::Format::RG16_SINT,
    nri::Format::RG16_SFLOAT,
    nri::Format::RGBA16_UNORM,
    nri::Format::RGBA16_SNORM,
    nri::Format::RGBA16_UINT,
    nri::Format::RGBA16_SINT,
    nri::Format::RGBA16_SFLOAT,
    nri::Format::R32_UINT,
    nri::Format::R32_SINT,
    nri::Format::R32_SFLOAT,
    nri::Format::RG32_UINT,
    nri::Format::RG32_SINT,
    nri::Format::RG32_SFLOAT,
    nri::Format::RGB32_UINT,
    nri::Format::RGB32_SINT,
    nri::Format::RGB32_SFLOAT,
    nri::Format::RGBA32_UINT,
    nri::Format::RGBA32_SINT,
    nri::Format::RGBA32_SFLOAT,
    nri::Format::R10_G10_B10_A2_UNORM,
    nri::Format::R10_G10_B10_A2_UINT,
    nri::Format::R11_G11_B10_UFLOAT,
    nri::Format::R9_G9_B9_E5_UFLOAT,
};

static inline uint16_t DivideUp(uint32_t x, uint16_t y) {
    return uint16_t((x + y - 1) / y);
}

static inline nri::Format GetNriFormat(Format format) {
    return g_NrdFormatToNri[(uint32_t)format];
}

static inline uint64_t CreateDescriptorKey(uint64_t texture, bool isStorage) {
    uint64_t key = uint64_t(isStorage ? 1 : 0) << 63ull;
    key |= texture & ((1ull << 63ull) - 1);

    return key;
}

template <typename T, typename A>
constexpr T Align(const T& size, A alignment) {
    return T(((size + alignment - 1) / alignment) * alignment);
}

Result Integration::Recreate(const IntegrationCreationDesc& integrationDesc, const InstanceCreationDesc& instanceDesc, nri::Device* device) {
    NRD_INTEGRATION_ASSERT(!integrationDesc.promoteFloat16to32 || !integrationDesc.demoteFloat32to16, "Can't be 'true' for both");
    NRD_INTEGRATION_ASSERT(integrationDesc.queuedFrameNum, "Can't be 0");

    if (m_SkipDestroy)
        m_SkipDestroy = false;
    else
        Destroy();

#ifdef NRD_INTEGRATION_DEBUG_LOGGING
    char filename[128];
    snprintf(filename, sizeof(filename), "NRD-%s.log", integrationDesc.name);
    m_Log = fopen(filename, "w");
    if (m_Log)
        fprintf(m_Log, "Recreating with resource size = %u x %u\n", integrationDesc.resourceWidth, integrationDesc.resourceHeight);
#endif

    if (nri::nriGetInterface(*device, NRI_INTERFACE(nri::CoreInterface), &m_iCore) != nri::Result::SUCCESS) {
        NRD_INTEGRATION_ASSERT(false, "'nriGetInterface(CoreInterface)' failed!");
        return Result::FAILURE;
    }

    const nri::DeviceDesc& deviceDesc = m_iCore.GetDeviceDesc(*device);
    if (deviceDesc.nriVersion != NRI_VERSION) {
        NRD_INTEGRATION_ASSERT(false, "NRI version mismatch detected!");
        return Result::FAILURE;
    }

    const LibraryDesc& libraryDesc = GetLibraryDesc();
    if (libraryDesc.versionMajor != NRD_VERSION_MAJOR || libraryDesc.versionMinor != NRD_VERSION_MINOR) {
        NRD_INTEGRATION_ASSERT(false, "NRD version mismatch detected!");
        return Result::FAILURE;
    }

    m_Desc = integrationDesc;
    m_Device = device;

    Result result = CreateInstance(instanceDesc, m_Instance);
    if (result == Result::SUCCESS) {
        result = _CreateResources() ? Result::SUCCESS : Result::FAILURE;
        if (result == Result::SUCCESS)
            result = RecreatePipelines() ? Result::SUCCESS : Result::FAILURE;
    }

    if (result != Result::SUCCESS)
        Destroy();

    return result;
}

#ifdef NRI_WRAPPER_D3D11
Result Integration::RecreateD3D11(const IntegrationCreationDesc& nrdIntegrationDesc, const InstanceCreationDesc& instanceCreationDesc, const nri::DeviceCreationD3D11Desc& deviceCreationD3D11Desc) {
    Destroy();

    if (nri::nriCreateDeviceFromD3D11Device(deviceCreationD3D11Desc, m_Device) != nri::Result::SUCCESS)
        return Result::FAILURE;

    if (nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::WrapperD3D11Interface), &m_iWrapperD3D11) != nri::Result::SUCCESS)
        return Result::FAILURE;

    m_Wrapped = nri::GraphicsAPI::D3D11;
    m_SkipDestroy = true;

    return Recreate(nrdIntegrationDesc, instanceCreationDesc, m_Device);
}
#endif

#ifdef NRI_WRAPPER_D3D12
Result Integration::RecreateD3D12(const IntegrationCreationDesc& nrdIntegrationDesc, const InstanceCreationDesc& instanceCreationDesc, const nri::DeviceCreationD3D12Desc& deviceCreationD3D12Desc) {
    Destroy();

    if (nri::nriCreateDeviceFromD3D12Device(deviceCreationD3D12Desc, m_Device) != nri::Result::SUCCESS)
        return Result::FAILURE;

    if (nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::WrapperD3D12Interface), &m_iWrapperD3D12) != nri::Result::SUCCESS)
        return Result::FAILURE;

    m_Wrapped = nri::GraphicsAPI::D3D12;
    m_SkipDestroy = true;

    return Recreate(nrdIntegrationDesc, instanceCreationDesc, m_Device);
}
#endif

#ifdef NRI_WRAPPER_VK
Result Integration::RecreateVK(const IntegrationCreationDesc& nrdIntegrationDesc, const InstanceCreationDesc& instanceCreationDesc, const nri::DeviceCreationVKDesc& deviceCreationVKDesc) {
    Destroy();

    if (nri::nriCreateDeviceFromVKDevice(deviceCreationVKDesc, m_Device) != nri::Result::SUCCESS)
        return Result::FAILURE;

    if (nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::WrapperVKInterface), &m_iWrapperVK) != nri::Result::SUCCESS)
        return Result::FAILURE;

    m_Wrapped = nri::GraphicsAPI::VK;
    m_SkipDestroy = true;

    return Recreate(nrdIntegrationDesc, instanceCreationDesc, m_Device);
}
#endif

bool Integration::RecreatePipelines() {
    _WaitForIdle();

#ifdef PROJECT_NAME
    utils::ShaderCodeStorage shaderCodeStorage;
#endif

    // Destroy old
    for (nri::Pipeline* pipeline : m_Pipelines)
        m_iCore.DestroyPipeline(*pipeline);
    m_Pipelines.clear();

    // Create new
    const InstanceDesc& instanceDesc = GetInstanceDesc(*m_Instance);
    const nri::DeviceDesc& deviceDesc = m_iCore.GetDeviceDesc(*m_Device);

    for (uint32_t i = 0; i < instanceDesc.pipelinesNum; i++) {
        const PipelineDesc& nrdPipelineDesc = instanceDesc.pipelines[i];
        const ComputeShaderDesc& nrdComputeShader = (&nrdPipelineDesc.computeShaderDXBC)[std::max((int32_t)deviceDesc.graphicsAPI - 1, 0)];

        nri::ShaderDesc computeShader = {};
#ifdef PROJECT_NAME
        if (nrdComputeShader.bytecode && !m_ReloadShaders) {
#endif
            computeShader.bytecode = nrdComputeShader.bytecode;
            computeShader.size = nrdComputeShader.size;
            computeShader.entryPointName = nrdPipelineDesc.shaderEntryPointName;
            computeShader.stage = nri::StageBits::COMPUTE_SHADER;
#ifdef PROJECT_NAME
        } else
            computeShader = utils::LoadShader(deviceDesc.graphicsAPI, nrdPipelineDesc.shaderFileName, shaderCodeStorage, nrdPipelineDesc.shaderEntryPointName);
#endif

        nri::ComputePipelineDesc pipelineDesc = {};
        pipelineDesc.pipelineLayout = m_PipelineLayout;
        pipelineDesc.shader = computeShader;

        nri::Pipeline* pipeline = nullptr;
        NRD_INTEGRATION_RETURN_FALSE_ON_FAILURE(m_iCore.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    m_ReloadShaders = true;

    return true;
}

bool Integration::_CreateResources() {
    const InstanceDesc& instanceDesc = GetInstanceDesc(*m_Instance);
    const nri::DeviceDesc& deviceDesc = m_iCore.GetDeviceDesc(*m_Device);
    const uint32_t poolSize = instanceDesc.permanentPoolSize + instanceDesc.transientPoolSize;

    { // Texture pool
        // No reallocation, please!
        m_TexturePool.resize(poolSize);

        for (uint32_t i = 0; i < poolSize; i++) {
            // Create NRI texture
            char name[128];
            nri::Texture* texture = nullptr;
            const TextureDesc& nrdTextureDesc = (i < instanceDesc.permanentPoolSize) ? instanceDesc.permanentPool[i] : instanceDesc.transientPool[i - instanceDesc.permanentPoolSize];
            {
                nri::Format format = GetNriFormat(nrdTextureDesc.format);
                if (m_Desc.promoteFloat16to32) {
                    if (format == nri::Format::R16_SFLOAT)
                        format = nri::Format::R32_SFLOAT;
                    else if (format == nri::Format::RG16_SFLOAT)
                        format = nri::Format::RG32_SFLOAT;
                    else if (format == nri::Format::RGBA16_SFLOAT)
                        format = nri::Format::RGBA32_SFLOAT;
                } else if (m_Desc.demoteFloat32to16) {
                    if (format == nri::Format::R32_SFLOAT)
                        format = nri::Format::R16_SFLOAT;
                    else if (format == nri::Format::RG32_SFLOAT)
                        format = nri::Format::RG16_SFLOAT;
                    else if (format == nri::Format::RGBA32_SFLOAT)
                        format = nri::Format::RGBA16_SFLOAT;
                }

                uint16_t w = DivideUp(m_Desc.resourceWidth, nrdTextureDesc.downsampleFactor);
                uint16_t h = DivideUp(m_Desc.resourceHeight, nrdTextureDesc.downsampleFactor);

                nri::TextureDesc textureDesc = {};
                textureDesc.type = nri::TextureType::TEXTURE_2D;
                textureDesc.usage = nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE;
                textureDesc.format = format;
                textureDesc.width = w;
                textureDesc.height = h;

                NRD_INTEGRATION_RETURN_FALSE_ON_FAILURE(m_iCore.CreateTexture(*m_Device, textureDesc, texture));

                if (i < instanceDesc.permanentPoolSize)
                    snprintf(name, sizeof(name), "%s::P(%u)", m_Desc.name, i);
                else
                    snprintf(name, sizeof(name), "%s::T(%u)", m_Desc.name, i - instanceDesc.permanentPoolSize);
                m_iCore.SetDebugName(texture, name);
            }

            { // Construct NRD texture
                Resource& resource = m_TexturePool[i];
                resource.nri.texture = texture;
                resource.state = {nri::AccessBits::UNKNOWN, nri::Layout::UNKNOWN};
            }

            { // Adjust memory usage
                nri::MemoryDesc memoryDesc = {};
                m_iCore.GetTextureMemoryDesc(*texture, nri::MemoryLocation::DEVICE, memoryDesc);

                if (i < instanceDesc.permanentPoolSize)
                    m_PermanentPoolSize += memoryDesc.size;
                else
                    m_TransientPoolSize += memoryDesc.size;
            }

#ifdef NRD_INTEGRATION_DEBUG_LOGGING
            if (m_Log)
                fprintf(m_Log, "%s\n\tformat=%u downsampleFactor=%u\n", name, nrdTextureDesc.format, nrdTextureDesc.downsampleFactor);
        }

        if (m_Log)
            fprintf(m_Log, "%.1f Mb (permanent), %.1f Mb (transient)\n\n", double(m_PermanentPoolSize) / (1024.0f * 1024.0f), double(m_TransientPoolSize) / (1024.0f * 1024.0f));
#else
        }
#endif
    }

    { // Samplers
        for (uint32_t i = 0; i < instanceDesc.samplersNum; i++) {
            Sampler nrdSampler = instanceDesc.samplers[i];

            nri::SamplerDesc samplerDesc = {};
            samplerDesc.addressModes = {nri::AddressMode::CLAMP_TO_EDGE, nri::AddressMode::CLAMP_TO_EDGE};
            samplerDesc.filters.min = nrdSampler == Sampler::NEAREST_CLAMP ? nri::Filter::NEAREST : nri::Filter::LINEAR;
            samplerDesc.filters.mag = nrdSampler == Sampler::NEAREST_CLAMP ? nri::Filter::NEAREST : nri::Filter::LINEAR;

            nri::Descriptor* descriptor = nullptr;
            NRD_INTEGRATION_RETURN_FALSE_ON_FAILURE(m_iCore.CreateSampler(*m_Device, samplerDesc, descriptor));
            m_Samplers.push_back(descriptor);
        }
    }

    { // Constant buffer
        m_ConstantBufferViewSize = Align(instanceDesc.constantBufferMaxDataSize, deviceDesc.memoryAlignment.constantBufferOffset);
        m_ConstantBufferSize = uint64_t(m_ConstantBufferViewSize) * instanceDesc.descriptorPoolDesc.setsMaxNum * m_Desc.queuedFrameNum;

        nri::BufferDesc bufferDesc = {};
        bufferDesc.size = m_ConstantBufferSize;
        bufferDesc.usage = nri::BufferUsageBits::CONSTANT_BUFFER;
        NRD_INTEGRATION_RETURN_FALSE_ON_FAILURE(m_iCore.CreateBuffer(*m_Device, bufferDesc, m_ConstantBuffer));
    }

    { // Bind resources to memory
        nri::HelperInterface iHelper = {};
        NRD_INTEGRATION_RETURN_FALSE_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::HelperInterface), &iHelper));

        std::vector<nri::Texture*> textures(m_TexturePool.size(), nullptr);
        for (size_t i = 0; i < m_TexturePool.size(); i++)
            textures[i] = m_TexturePool[i].nri.texture;

        nri::ResourceGroupDesc resourceGroupDesc = {};
        resourceGroupDesc.memoryLocation = nri::MemoryLocation::DEVICE;
        resourceGroupDesc.textureNum = (uint32_t)textures.size();
        resourceGroupDesc.textures = textures.data();

        size_t baseAllocation = m_MemoryAllocations.size();
        size_t allocationNum = iHelper.CalculateAllocationNumber(*m_Device, resourceGroupDesc);
        m_MemoryAllocations.resize(baseAllocation + allocationNum, nullptr);
        NRD_INTEGRATION_RETURN_FALSE_ON_FAILURE(iHelper.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data() + baseAllocation));

        resourceGroupDesc = {};
        resourceGroupDesc.memoryLocation = nri::MemoryLocation::HOST_UPLOAD;
        resourceGroupDesc.bufferNum = 1;
        resourceGroupDesc.buffers = &m_ConstantBuffer;

        baseAllocation = m_MemoryAllocations.size();
        m_MemoryAllocations.resize(baseAllocation + 1, nullptr);
        NRD_INTEGRATION_RETURN_FALSE_ON_FAILURE(iHelper.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data() + baseAllocation));
    }

    { // Constant buffer view
        nri::BufferViewDesc constantBufferViewDesc = {};
        constantBufferViewDesc.viewType = nri::BufferViewType::CONSTANT;
        constantBufferViewDesc.buffer = m_ConstantBuffer;
        constantBufferViewDesc.size = m_ConstantBufferViewSize;
        NRD_INTEGRATION_RETURN_FALSE_ON_FAILURE(m_iCore.CreateBufferView(constantBufferViewDesc, m_ConstantBufferView));
    }

    // Pipeline layout
    nri::DescriptorRangeDesc descriptorRanges[3] = {};
    {
        // SPIRV binding offsets
        uint32_t constantBufferOffset = 0;
        uint32_t samplerOffset = 0;
        uint32_t textureOffset = 0;
        uint32_t storageTextureOffset = 0;

        if (deviceDesc.graphicsAPI == nri::GraphicsAPI::VK) {
            const LibraryDesc& nrdLibraryDesc = GetLibraryDesc();
            constantBufferOffset = nrdLibraryDesc.spirvBindingOffsets.constantBufferOffset;
            samplerOffset = nrdLibraryDesc.spirvBindingOffsets.samplerOffset;
            textureOffset = nrdLibraryDesc.spirvBindingOffsets.textureOffset;
            storageTextureOffset = nrdLibraryDesc.spirvBindingOffsets.storageTextureAndBufferOffset;
        }

        // Descriptor ranges
        descriptorRanges[RANGE_TEXTURES].baseRegisterIndex = textureOffset + instanceDesc.resourcesBaseRegisterIndex;
        descriptorRanges[RANGE_TEXTURES].descriptorNum = instanceDesc.descriptorPoolDesc.perSetTexturesMaxNum;
        descriptorRanges[RANGE_TEXTURES].descriptorType = nri::DescriptorType::TEXTURE;
        descriptorRanges[RANGE_TEXTURES].shaderStages = nri::StageBits::COMPUTE_SHADER;
        descriptorRanges[RANGE_TEXTURES].flags = nri::DescriptorRangeBits::PARTIALLY_BOUND;

        descriptorRanges[RANGE_STORAGES].baseRegisterIndex = storageTextureOffset + instanceDesc.resourcesBaseRegisterIndex;
        descriptorRanges[RANGE_STORAGES].descriptorNum = instanceDesc.descriptorPoolDesc.perSetStorageTexturesMaxNum;
        descriptorRanges[RANGE_STORAGES].descriptorType = nri::DescriptorType::STORAGE_TEXTURE;
        descriptorRanges[RANGE_STORAGES].shaderStages = nri::StageBits::COMPUTE_SHADER;
        descriptorRanges[RANGE_STORAGES].flags = nri::DescriptorRangeBits::PARTIALLY_BOUND;

        descriptorRanges[RANGE_SAMPLERS].baseRegisterIndex = samplerOffset + instanceDesc.samplersBaseRegisterIndex;
        descriptorRanges[RANGE_SAMPLERS].descriptorNum = instanceDesc.samplersNum;
        descriptorRanges[RANGE_SAMPLERS].descriptorType = nri::DescriptorType::SAMPLER;
        descriptorRanges[RANGE_SAMPLERS].shaderStages = nri::StageBits::COMPUTE_SHADER;

        // Descriptor sets
        const nri::DynamicConstantBufferDesc dynamicConstantBufferDesc = {constantBufferOffset + instanceDesc.constantBufferRegisterIndex, nri::StageBits::COMPUTE_SHADER};

        nri::DescriptorSetDesc descriptorSetDescs[2] = {};

        descriptorSetDescs[SET_ALL].registerSpace = instanceDesc.constantBufferAndResourcesSpaceIndex;
        descriptorSetDescs[SET_ALL].dynamicConstantBuffers = &dynamicConstantBufferDesc;
        descriptorSetDescs[SET_ALL].dynamicConstantBufferNum = 1;
        descriptorSetDescs[SET_ALL].ranges = descriptorRanges;
        descriptorSetDescs[SET_ALL].rangeNum = 3;

        if (instanceDesc.samplersInSeparateSet) {
            descriptorSetDescs[SET_ALL].rangeNum = 2;

            descriptorSetDescs[SET_SAMPLERS].registerSpace = instanceDesc.samplersSpaceIndex;
            descriptorSetDescs[SET_SAMPLERS].ranges = &descriptorRanges[RANGE_SAMPLERS];
            descriptorSetDescs[SET_SAMPLERS].rangeNum = 1;
        }

        // Pipeline layout
        nri::PipelineLayoutDesc pipelineLayoutDesc = {};
        pipelineLayoutDesc.descriptorSetNum = instanceDesc.samplersInSeparateSet ? 2 : 1;
        pipelineLayoutDesc.descriptorSets = descriptorSetDescs;
        pipelineLayoutDesc.shaderStages = nri::StageBits::COMPUTE_SHADER;
        pipelineLayoutDesc.flags = nri::PipelineLayoutBits::IGNORE_GLOBAL_SPIRV_OFFSETS;

        NRD_INTEGRATION_RETURN_FALSE_ON_FAILURE(m_iCore.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, m_PipelineLayout));
    }

    { // Descriptor pools
        uint32_t setMaxNum = instanceDesc.descriptorPoolDesc.setsMaxNum;

        nri::DescriptorPoolDesc descriptorPoolDesc = {};
        descriptorPoolDesc.descriptorSetMaxNum = setMaxNum;
        descriptorPoolDesc.dynamicConstantBufferMaxNum = setMaxNum;
        descriptorPoolDesc.textureMaxNum = setMaxNum * descriptorRanges[RANGE_TEXTURES].descriptorNum;
        descriptorPoolDesc.storageTextureMaxNum = setMaxNum * descriptorRanges[RANGE_STORAGES].descriptorNum;
        descriptorPoolDesc.samplerMaxNum = (instanceDesc.samplersInSeparateSet ? 1 : setMaxNum) * descriptorRanges[RANGE_SAMPLERS].descriptorNum;

        for (uint32_t i = 0; i < m_Desc.queuedFrameNum; i++) {
            nri::DescriptorPool* descriptorPool = nullptr;
            NRD_INTEGRATION_RETURN_FALSE_ON_FAILURE(m_iCore.CreateDescriptorPool(*m_Device, descriptorPoolDesc, descriptorPool));
            m_DescriptorPools.push_back(descriptorPool);

            m_DescriptorSetSamplers.push_back(nullptr);
            m_DescriptorsInFlight.push_back({});
        }
    }

#ifdef NRD_INTEGRATION_DEBUG_LOGGING
    if (m_Log)
        fflush(m_Log);
#endif

    return true;
}

void Integration::NewFrame() {
    NRD_INTEGRATION_ASSERT(m_Instance, "Uninitialized! Did you forget to call 'Recreate'?");

    // Must be here since the initial value is "-1", otherwise "descriptorPool[0]" will be used twice on the 1st and 2nd frames
    m_FrameIndex++;

#ifdef NRD_INTEGRATION_DEBUG_LOGGING
    if (m_Log) {
        fflush(m_Log);
        fprintf(m_Log, "Frame %u ==============================================================================\n\n", m_FrameIndex);
    }
#endif

    // Current descriptor pool index
    m_DescriptorPoolIndex = m_FrameIndex % m_Desc.queuedFrameNum;

    // Reset descriptor pool and samplers (since they are allocated from it)
    nri::DescriptorPool* descriptorPool = m_DescriptorPools[m_DescriptorPoolIndex];
    m_iCore.ResetDescriptorPool(*descriptorPool);

    m_DescriptorSetSamplers[m_DescriptorPoolIndex] = nullptr;

    // Referenced by the GPU descriptors can't be destroyed...
    if (!m_Desc.enableWholeLifetimeDescriptorCaching) {
        auto& descriptors = m_DescriptorsInFlight[m_DescriptorPoolIndex];

        for (const auto& descriptor : descriptors)
            m_iCore.DestroyDescriptor(*descriptor);

#ifdef NRD_INTEGRATION_DEBUG_LOGGING
        if (m_Log)
            fprintf(m_Log, "Destroyed %u cached descriptors (queued frame = %u, totalNum = 0)\n", (uint32_t)descriptors.size(), m_DescriptorPoolIndex);
#endif

        descriptors.clear();
    }

    m_PrevFrameIndexFromSettings++;
}

Result Integration::SetCommonSettings(const CommonSettings& commonSettings) {
    NRD_INTEGRATION_ASSERT(m_Instance, "Uninitialized! Did you forget to call 'Recreate'?");
    NRD_INTEGRATION_ASSERT(commonSettings.resourceSize[0] == commonSettings.resourceSizePrev[0]
            && commonSettings.resourceSize[1] == commonSettings.resourceSizePrev[1]
            && commonSettings.resourceSize[0] == m_Desc.resourceWidth && commonSettings.resourceSize[1] == m_Desc.resourceHeight,
        "NRD integration preallocates resources statically: DRS is only supported via 'rectSize / rectSizePrev'");

    Result result = nrd::SetCommonSettings(*m_Instance, commonSettings);
    NRD_INTEGRATION_ASSERT(result == Result::SUCCESS, "SetCommonSettings() failed!");

    if (m_FrameIndex == 0 || commonSettings.accumulationMode != AccumulationMode::CONTINUE)
        m_PrevFrameIndexFromSettings = commonSettings.frameIndex;
    else
        NRD_INTEGRATION_ASSERT(m_PrevFrameIndexFromSettings == commonSettings.frameIndex, "'frameIndex' must be incremented by 1 on each frame");

    return result;
}

Result Integration::SetDenoiserSettings(Identifier denoiser, const void* denoiserSettings) {
    NRD_INTEGRATION_ASSERT(m_Instance, "Uninitialized! Did you forget to call 'Recreate'?");

    Result result = nrd::SetDenoiserSettings(*m_Instance, denoiser, denoiserSettings);
    NRD_INTEGRATION_ASSERT(result == Result::SUCCESS, "SetDenoiserSettings() failed!");

    return result;
}

void Integration::Denoise(const Identifier* denoisers, uint32_t denoisersNum, nri::CommandBuffer& commandBuffer, ResourceSnapshot& resourceSnapshot) {
    NRD_INTEGRATION_ASSERT(m_Instance, "Uninitialized! Did you forget to call 'Recreate'?");

    // Save initial state
    nri::AccessLayoutStage* initialStates = (nri::AccessLayoutStage*)alloca(sizeof(nri::AccessLayoutStage) * resourceSnapshot.uniqueNum);
    for (size_t i = 0; i < resourceSnapshot.uniqueNum; i++)
        initialStates[i] = resourceSnapshot.unique[i].state;

    // One time sanity check
    if (m_FrameIndex == 0) {
        const nri::Texture* normalRoughnessTexture = resourceSnapshot.slots[(size_t)ResourceType::IN_NORMAL_ROUGHNESS]->nri.texture;
        const nri::TextureDesc& normalRoughnessDesc = m_iCore.GetTextureDesc(*normalRoughnessTexture);
        const LibraryDesc& nrdLibraryDesc = GetLibraryDesc();

        bool isNormalRoughnessFormatValid = false;
        switch (nrdLibraryDesc.normalEncoding) {
            case NormalEncoding::RGBA8_UNORM:
                isNormalRoughnessFormatValid = normalRoughnessDesc.format == nri::Format::RGBA8_UNORM;
                break;
            case NormalEncoding::RGBA8_SNORM:
                isNormalRoughnessFormatValid = normalRoughnessDesc.format == nri::Format::RGBA8_SNORM;
                break;
            case NormalEncoding::R10_G10_B10_A2_UNORM:
                isNormalRoughnessFormatValid = normalRoughnessDesc.format == nri::Format::R10_G10_B10_A2_UNORM;
                break;
            case NormalEncoding::RGBA16_UNORM:
                isNormalRoughnessFormatValid = normalRoughnessDesc.format == nri::Format::RGBA16_UNORM;
                break;
            case NormalEncoding::RGBA16_SNORM:
                isNormalRoughnessFormatValid = normalRoughnessDesc.format == nri::Format::RGBA16_SNORM || normalRoughnessDesc.format == nri::Format::RGBA16_SFLOAT || normalRoughnessDesc.format == nri::Format::RGBA32_SFLOAT;
                break;
        }

        NRD_INTEGRATION_ASSERT(isNormalRoughnessFormatValid, "IN_NORMAL_ROUGHNESS format doesn't match NRD normal encoding");
    }

    // Retrieve dispatches
    const DispatchDesc* dispatchDescs = nullptr;
    uint32_t dispatchDescsNum = 0;
    GetComputeDispatches(*m_Instance, denoisers, denoisersNum, dispatchDescs, dispatchDescsNum);

    // Even if descriptor caching is disabled it's better to cache descriptors inside a single "Denoise" call
    if (!m_Desc.enableWholeLifetimeDescriptorCaching)
        m_CachedDescriptors.clear();

    // Set descriptor pool
    nri::DescriptorPool* descriptorPool = m_DescriptorPools[m_DescriptorPoolIndex];
    m_iCore.CmdSetDescriptorPool(commandBuffer, *descriptorPool);

    // Invoke dispatches
    constexpr uint32_t lawnGreen = 0xFF7CFC00;
    constexpr uint32_t limeGreen = 0xFF32CD32;

    m_iCore.CmdSetPipelineLayout(commandBuffer, *m_PipelineLayout);

    for (uint32_t i = 0; i < dispatchDescsNum; i++) {
        const DispatchDesc& dispatchDesc = dispatchDescs[i];
        m_iCore.CmdBeginAnnotation(commandBuffer, dispatchDesc.name, (i & 0x1) ? lawnGreen : limeGreen);

        _Dispatch(commandBuffer, *descriptorPool, dispatchDesc, resourceSnapshot);

        m_iCore.CmdEndAnnotation(commandBuffer);
    }

    // Restore state
    if (resourceSnapshot.restoreInitialState) {
        nri::TextureBarrierDesc* textureBarriers = (nri::TextureBarrierDesc*)alloca(sizeof(nri::TextureBarrierDesc) * resourceSnapshot.uniqueNum);
        uint32_t textureBarrierNum = 0;

        for (size_t i = 0; i < resourceSnapshot.uniqueNum; i++) {
            Resource& resource = resourceSnapshot.unique[i];
            const nri::AccessLayoutStage& initialState = initialStates[i];

            bool isDifferent = resource.state.access != initialState.access || resource.state.layout != initialState.layout;
            bool isUnknown = initialState.access == nri::AccessBits::UNKNOWN || initialState.layout == nri::Layout::UNKNOWN;

            if (resource.nri.texture && isDifferent && !isUnknown) {
                nri::TextureBarrierDesc& barrier = textureBarriers[textureBarrierNum++];

                barrier = {};
                barrier.texture = resource.nri.texture;
                barrier.before = resource.state;
                barrier.after = initialState;

                resource.state = initialState;
            }
        }

        if (textureBarrierNum) {
            nri::BarrierGroupDesc transitionBarriers = {};
            transitionBarriers.textures = textureBarriers;
            transitionBarriers.textureNum = textureBarrierNum;

            m_iCore.CmdBarrier(commandBuffer, transitionBarriers);
        }
    }
}

#ifdef NRI_WRAPPER_D3D11
void Integration::DenoiseD3D11(const Identifier* denoisers, uint32_t denoisersNum, const nri::CommandBufferD3D11Desc& commandBufferD3D11Desc, ResourceSnapshot& resourceSnapshot) {
    NRD_INTEGRATION_ASSERT(m_Wrapped == nri::GraphicsAPI::D3D11, "GAPI mismatch");

    // Wrap
    for (size_t i = 0; i < resourceSnapshot.uniqueNum; i++) {
        Resource& resource = resourceSnapshot.unique[i];

        nri::TextureD3D11Desc textureDesc = {};
        textureDesc.d3d11Resource = resource.d3d11.resource;
        textureDesc.format = resource.d3d11.format;

        nri::Result result = m_iWrapperD3D11.CreateTextureD3D11(*m_Device, textureDesc, resource.nri.texture);
        NRD_INTEGRATION_ASSERT(result == nri::Result::SUCCESS, "CreateTextureD3D11() failed!");
    }

    nri::CommandBuffer* commandBuffer = nullptr;
    nri::Result result = m_iWrapperD3D11.CreateCommandBufferD3D11(*m_Device, commandBufferD3D11Desc, commandBuffer);
    NRD_INTEGRATION_ASSERT(result == nri::Result::SUCCESS, "CreateCommandBufferD3D11() failed!");

    // Denoise
    Denoise(denoisers, denoisersNum, *commandBuffer, resourceSnapshot);

    // Unwrap
    m_iCore.DestroyCommandBuffer(*commandBuffer);

    for (size_t i = 0; i < resourceSnapshot.uniqueNum; i++)
        m_iCore.DestroyTexture(*resourceSnapshot.unique[i].nri.texture);
}
#endif

#ifdef NRI_WRAPPER_D3D12
void Integration::DenoiseD3D12(const Identifier* denoisers, uint32_t denoisersNum, const nri::CommandBufferD3D12Desc& commandBufferD3D12Desc, ResourceSnapshot& resourceSnapshot) {
    NRD_INTEGRATION_ASSERT(m_Wrapped == nri::GraphicsAPI::D3D12, "GAPI mismatch");

    // Wrap
    for (size_t i = 0; i < resourceSnapshot.uniqueNum; i++) {
        Resource& resource = resourceSnapshot.unique[i];

        nri::TextureD3D12Desc textureDesc = {};
        textureDesc.d3d12Resource = resource.d3d12.resource;
        textureDesc.format = resource.d3d12.format;

        nri::Result result = m_iWrapperD3D12.CreateTextureD3D12(*m_Device, textureDesc, resource.nri.texture);
        NRD_INTEGRATION_ASSERT(result == nri::Result::SUCCESS, "CreateTextureD3D12() failed!");
    }

    nri::CommandBuffer* commandBuffer = nullptr;
    nri::Result result = m_iWrapperD3D12.CreateCommandBufferD3D12(*m_Device, commandBufferD3D12Desc, commandBuffer);
    NRD_INTEGRATION_ASSERT(result == nri::Result::SUCCESS, "CreateCommandBufferD3D12() failed!");

    // Denoise
    Denoise(denoisers, denoisersNum, *commandBuffer, resourceSnapshot);

    // Unwrap
    m_iCore.DestroyCommandBuffer(*commandBuffer);

    for (size_t i = 0; i < resourceSnapshot.uniqueNum; i++)
        m_iCore.DestroyTexture(*resourceSnapshot.unique[i].nri.texture);
}
#endif

#ifdef NRI_WRAPPER_VK
void Integration::DenoiseVK(const Identifier* denoisers, uint32_t denoisersNum, const nri::CommandBufferVKDesc& commandBufferVKDesc, ResourceSnapshot& resourceSnapshot) {
    NRD_INTEGRATION_ASSERT(m_Wrapped == nri::GraphicsAPI::VK, "GAPI mismatch");

    // Wrap
    for (size_t i = 0; i < resourceSnapshot.uniqueNum; i++) {
        Resource& resource = resourceSnapshot.unique[i];

        nri::TextureVKDesc textureDesc = {};
        textureDesc.vkImage = resource.vk.image;
        textureDesc.vkFormat = resource.vk.format;
        textureDesc.vkImageType = 1; // VK_IMAGE_TYPE_2D
        textureDesc.width = m_Desc.resourceWidth;
        textureDesc.height = m_Desc.resourceHeight;
        textureDesc.depth = 1;
        textureDesc.mipNum = 1;
        textureDesc.layerNum = 1;
        textureDesc.sampleNum = 1;

        nri::Result result = m_iWrapperVK.CreateTextureVK(*m_Device, textureDesc, resource.nri.texture);
        NRD_INTEGRATION_ASSERT(result == nri::Result::SUCCESS, "CreateTextureVK() failed!");
    }

    nri::CommandBuffer* commandBuffer = nullptr;
    nri::Result result = m_iWrapperVK.CreateCommandBufferVK(*m_Device, commandBufferVKDesc, commandBuffer);
    NRD_INTEGRATION_ASSERT(result == nri::Result::SUCCESS, "CreateCommandBufferVK() failed!");

    // Denoise
    Denoise(denoisers, denoisersNum, *commandBuffer, resourceSnapshot);

    // Unwrap
    m_iCore.DestroyCommandBuffer(*commandBuffer);

    for (size_t i = 0; i < resourceSnapshot.uniqueNum; i++)
        m_iCore.DestroyTexture(*resourceSnapshot.unique[i].nri.texture);
}
#endif

void Integration::_Dispatch(nri::CommandBuffer& commandBuffer, nri::DescriptorPool& descriptorPool, const DispatchDesc& dispatchDesc, ResourceSnapshot& resourceSnapshot) {
    const InstanceDesc& instanceDesc = GetInstanceDesc(*m_Instance);
    const PipelineDesc& pipelineDesc = instanceDesc.pipelines[dispatchDesc.pipelineIndex];

    nri::Descriptor** descriptors = (nri::Descriptor**)alloca(sizeof(nri::Descriptor*) * dispatchDesc.resourcesNum);
    memset(descriptors, 0, sizeof(nri::Descriptor*) * dispatchDesc.resourcesNum);

    nri::TextureBarrierDesc* transitions = (nri::TextureBarrierDesc*)alloca(sizeof(nri::TextureBarrierDesc) * dispatchDesc.resourcesNum);
    memset(transitions, 0, sizeof(nri::TextureBarrierDesc) * dispatchDesc.resourcesNum);

    nri::BarrierGroupDesc transitionBarriers = {};
    transitionBarriers.textures = transitions;

    std::array<nri::DescriptorRangeUpdateDesc, 3> descriptorRanges = {};
    descriptorRanges[RANGE_SAMPLERS] = {m_Samplers.data(), instanceDesc.samplersNum, 0};

    uint32_t createdDescriptorNum = 0;

    { // Fill descriptors and ranges
        uint32_t n = 0;
        for (uint32_t i = 0; i < pipelineDesc.resourceRangesNum; i++) {
            const ResourceRangeDesc& resourceRange = pipelineDesc.resourceRanges[i];
            const bool isStorage = resourceRange.descriptorType == DescriptorType::STORAGE_TEXTURE;

            uint32_t rangeIndex = isStorage ? RANGE_STORAGES : RANGE_TEXTURES;
            descriptorRanges[rangeIndex].descriptors = descriptors + n;
            descriptorRanges[rangeIndex].descriptorNum = resourceRange.descriptorsNum;

            for (uint32_t j = 0; j < resourceRange.descriptorsNum; j++) {
                const ResourceDesc& resourceDesc = dispatchDesc.resources[n];

                // Get resource
                Resource* resource = nullptr;
                if (resourceDesc.type == ResourceType::TRANSIENT_POOL)
                    resource = &m_TexturePool[resourceDesc.indexInPool + instanceDesc.permanentPoolSize];
                else if (resourceDesc.type == ResourceType::PERMANENT_POOL)
                    resource = &m_TexturePool[resourceDesc.indexInPool];
                else {
                    resource = resourceSnapshot.slots[(uint32_t)resourceDesc.type];
                    NRD_INTEGRATION_ASSERT(resource->nri.texture, "invalid entry!");
                }

                // Prepare barrier
                nri::AccessLayoutStage after = {};
                if (resourceDesc.descriptorType == DescriptorType::TEXTURE)
                    after = {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE, nri::StageBits::COMPUTE_SHADER};
                else
                    after = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER};

                bool isStateChanged = after.access != resource->state.access || after.layout != resource->state.layout;
                bool isStorageBarrier = after.access == nri::AccessBits::SHADER_RESOURCE_STORAGE && resource->state.access == nri::AccessBits::SHADER_RESOURCE_STORAGE;
                if (isStateChanged || isStorageBarrier) {
                    nri::TextureBarrierDesc& barrier = transitions[transitionBarriers.textureNum++];

                    barrier = {};
                    barrier.texture = resource->nri.texture;
                    barrier.before = resource->state;
                    barrier.after = after;
                }

                resource->state = after;

                // Create descriptor
                uint64_t nativeObject = m_iCore.GetTextureNativeObject(*resource->nri.texture);
                uint64_t key = CreateDescriptorKey(nativeObject, isStorage);
                const auto& entry = m_CachedDescriptors.find(key);

                nri::Descriptor* descriptor = nullptr;
                if (entry == m_CachedDescriptors.end()) {
                    const nri::TextureDesc& textureDesc = m_iCore.GetTextureDesc(*resource->nri.texture);

                    nri::Texture2DViewDesc desc = {
                        resource->nri.texture,
                        isStorage ? nri::Texture2DViewType::SHADER_RESOURCE_STORAGE_2D : nri::Texture2DViewType::SHADER_RESOURCE_2D,
                        textureDesc.format,
                        0,
                        1,
                        0,
                        1,
                    };

                    nri::Result result = m_iCore.CreateTexture2DView(desc, descriptor);
                    NRD_INTEGRATION_ASSERT(result == nri::Result::SUCCESS, "CreateTexture2DView() failed!");

                    m_CachedDescriptors.insert(std::make_pair(key, descriptor));
                    m_DescriptorsInFlight[m_DescriptorPoolIndex].push_back(descriptor);

                    createdDescriptorNum++;
                } else
                    descriptor = entry->second;

                // Add descriptor to the range
                descriptors[n++] = descriptor;
            }
        }
    }

    // Allocate descriptor sets
    uint32_t descriptorSetNum = instanceDesc.samplersInSeparateSet ? 2 : 1;

    std::array<nri::DescriptorSet*, 2> descriptorSets = {};
    nri::Result result = m_iCore.AllocateDescriptorSets(descriptorPool, *m_PipelineLayout, SET_ALL, &descriptorSets[SET_ALL], 1, 0);
    NRD_INTEGRATION_ASSERT(result == nri::Result::SUCCESS, "AllocateDescriptorSets() failed!");

    // Update samplers
    if (instanceDesc.samplersInSeparateSet) {
        nri::DescriptorSet*& descriptorSetSamplers = m_DescriptorSetSamplers[m_DescriptorPoolIndex];
        if (!descriptorSetSamplers) {
            result = m_iCore.AllocateDescriptorSets(descriptorPool, *m_PipelineLayout, SET_SAMPLERS, &descriptorSetSamplers, 1, 0);
            NRD_INTEGRATION_ASSERT(result == nri::Result::SUCCESS, "AllocateDescriptorSets() failed!");

            m_iCore.UpdateDescriptorRanges(*descriptorSetSamplers, 0, 1, &descriptorRanges[RANGE_SAMPLERS]);
        }

        descriptorSets[SET_SAMPLERS] = descriptorSetSamplers;
    }

    // Update constants
    uint32_t dynamicConstantBufferOffset = m_ConstantBufferOffsetPrev;
    {
        // Stream data only if needed
        if (dispatchDesc.constantBufferDataSize && !dispatchDesc.constantBufferDataMatchesPreviousDispatch) {
            // Ring-buffer logic
            if (m_ConstantBufferOffset + m_ConstantBufferViewSize > m_ConstantBufferSize)
                m_ConstantBufferOffset = 0;

            dynamicConstantBufferOffset = m_ConstantBufferOffset;
            m_ConstantBufferOffset += m_ConstantBufferViewSize;

            // Upload CB data
            void* data = m_iCore.MapBuffer(*m_ConstantBuffer, dynamicConstantBufferOffset, dispatchDesc.constantBufferDataSize);
            if (data) {
                memcpy(data, dispatchDesc.constantBufferData, dispatchDesc.constantBufferDataSize);
                m_iCore.UnmapBuffer(*m_ConstantBuffer);
            }

            // Save previous offset for potential CB data reuse
            m_ConstantBufferOffsetPrev = dynamicConstantBufferOffset;
        }

        // Must be updated since it's declared in the shared pipeline layout
        m_iCore.UpdateDynamicConstantBuffers(*descriptorSets[SET_ALL], 0, 1, &m_ConstantBufferView);
    }

    // Update descriptor ranges
    uint32_t baseRange = pipelineDesc.resourceRangesNum == 1 ? RANGE_STORAGES : RANGE_TEXTURES;
    uint32_t rangeNum = pipelineDesc.resourceRangesNum + (instanceDesc.samplersInSeparateSet ? 0 : 1);

    m_iCore.UpdateDescriptorRanges(*descriptorSets[SET_ALL], baseRange, rangeNum, &descriptorRanges[baseRange]);

    // Rendering
    nri::Pipeline* pipeline = m_Pipelines[dispatchDesc.pipelineIndex];
    m_iCore.CmdSetPipeline(commandBuffer, *pipeline);

    for (uint32_t i = 0; i < descriptorSetNum; i++)
        m_iCore.CmdSetDescriptorSet(commandBuffer, i, *descriptorSets[i], i == SET_ALL ? &dynamicConstantBufferOffset : nullptr);

    m_iCore.CmdBarrier(commandBuffer, transitionBarriers);
    m_iCore.CmdDispatch(commandBuffer, {dispatchDesc.gridWidth, dispatchDesc.gridHeight, 1});

    // Debug logging
#ifdef NRD_INTEGRATION_DEBUG_LOGGING
    if (m_Log) {
        if (createdDescriptorNum)
            fprintf(m_Log, "Added %u cached descriptors (queued frame = %u, totalNum = %u)\n\n", createdDescriptorNum, m_DescriptorPoolIndex, (uint32_t)m_DescriptorsInFlight[m_DescriptorPoolIndex].size());

        fprintf(m_Log, "%c Pipeline #%u : %s\n\t", dispatchDesc.constantBufferDataMatchesPreviousDispatch ? ' ' : '!', dispatchDesc.pipelineIndex, dispatchDesc.name);
        for (uint32_t i = 0; i < dispatchDesc.resourcesNum; i++) {
            const ResourceDesc& r = dispatchDesc.resources[i];

            if (r.type == ResourceType::PERMANENT_POOL)
                fprintf(m_Log, "P(%u) ", r.indexInPool);
            else if (r.type == ResourceType::TRANSIENT_POOL)
                fprintf(m_Log, "T(%u) ", r.indexInPool);
            else {
                const char* s = GetResourceTypeString(r.type);
                fprintf(m_Log, "%s ", s);
            }
        }
        fprintf(m_Log, "\n\n");
    }
#else
    (void)createdDescriptorNum;
#endif
}

void Integration::DestroyCachedDescriptors() {
#ifdef NRD_INTEGRATION_DEBUG_LOGGING
    if (m_Log)
        fprintf(m_Log, "Destroy cached descriptors\n");
#endif

    if (!m_iCore.GetDeviceDesc)
        return;

    _WaitForIdle();

    for (auto& descriptors : m_DescriptorsInFlight) {
        for (const auto& descriptor : descriptors)
            m_iCore.DestroyDescriptor(*descriptor);

        descriptors.clear();
    }

    m_CachedDescriptors.clear();
}

void Integration::Destroy() {
#ifdef NRD_INTEGRATION_DEBUG_LOGGING
    if (m_Log)
        fprintf(m_Log, "Destroy\n");
#endif

    if (m_iCore.GetDeviceDesc) {
        _WaitForIdle();

        m_iCore.DestroyDescriptor(*m_ConstantBufferView);
        m_iCore.DestroyBuffer(*m_ConstantBuffer);
        m_iCore.DestroyPipelineLayout(*m_PipelineLayout);

        for (auto& descriptors : m_DescriptorsInFlight) {
            for (const auto& descriptor : descriptors)
                m_iCore.DestroyDescriptor(*descriptor);

            descriptors.clear();
        }

        for (const Resource& resource : m_TexturePool)
            m_iCore.DestroyTexture(*resource.nri.texture);

        for (nri::Descriptor* descriptor : m_Samplers)
            m_iCore.DestroyDescriptor(*descriptor);

        for (nri::Pipeline* pipeline : m_Pipelines)
            m_iCore.DestroyPipeline(*pipeline);

        for (nri::Memory* memory : m_MemoryAllocations)
            m_iCore.FreeMemory(*memory);

        for (nri::DescriptorPool* descriptorPool : m_DescriptorPools)
            m_iCore.DestroyDescriptorPool(*descriptorPool);

        if (m_Wrapped != nri::GraphicsAPI::NONE)
            nri::nriDestroyDevice(*m_Device);
    }

    if (m_Instance)
        DestroyInstance(*m_Instance);

    // Better keep in sync with the default values used by constructor
    m_TexturePool.clear();
    m_Pipelines.clear();
    m_MemoryAllocations.clear();
    m_Samplers.clear();
    m_DescriptorPools.clear();
    m_DescriptorSetSamplers.clear();
    m_DescriptorsInFlight.clear();
    m_CachedDescriptors.clear();
    m_Desc = {};
    m_iCore = {};
    m_Device = nullptr;
    m_Instance = nullptr;
    m_PermanentPoolSize = 0;
    m_TransientPoolSize = 0;
    m_ConstantBufferSize = 0;
    m_ConstantBufferViewSize = 0;
    m_ConstantBufferOffset = 0;
    m_ConstantBufferOffsetPrev = 0;
    m_DescriptorPoolIndex = 0;
    m_FrameIndex = uint32_t(-1);
    m_PrevFrameIndexFromSettings = 0;
    m_Wrapped = nri::GraphicsAPI::NONE;
    m_SkipDestroy = false;
    m_ReloadShaders = false;

#ifdef NRD_INTEGRATION_DEBUG_LOGGING
    if (m_Log)
        fclose(m_Log);
#endif
}

void Integration::_WaitForIdle() {
    if (!m_Desc.autoWaitForIdle)
        return;

    nri::HelperInterface iHelper = {};
    nri::Result result = nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::HelperInterface), &iHelper);
    NRD_INTEGRATION_ASSERT(result == nri::Result::SUCCESS, "nriGetInterface(HelperInterface) failed!");

    { // GRAPHICS
        nri::Queue* graphicsQueue = nullptr;
        result = m_iCore.GetQueue(*m_Device, nri::QueueType::GRAPHICS, 0, graphicsQueue);
        if (result == nri::Result::SUCCESS)
            iHelper.WaitForIdle(*graphicsQueue);
    }

    { // COMPUTE
        nri::Queue* computeQueue = nullptr;
        result = m_iCore.GetQueue(*m_Device, nri::QueueType::COMPUTE, 0, computeQueue);
        if (result == nri::Result::SUCCESS)
            iHelper.WaitForIdle(*computeQueue);
    }
}

} // namespace nrd
