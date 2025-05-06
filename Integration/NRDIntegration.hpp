/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "NRDIntegration.h"

#include <string.h> // strncpy
#ifdef _WIN32
    #include <malloc.h>
#else
    #include <alloca.h>
#endif

static_assert(NRD_VERSION_MAJOR >= 4 && NRD_VERSION_MINOR >= 14, "Unsupported NRD version!");
static_assert(NRI_VERSION_MAJOR >= 1 && NRI_VERSION_MINOR >= 165, "Unsupported NRI version!");

namespace nrd
{

constexpr std::array<nri::Format, (size_t)Format::MAX_NUM> g_NrdFormatToNri =
{
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

static inline uint16_t DivideUp(uint32_t x, uint16_t y)
{ return uint16_t((x + y - 1) / y); }

static inline nri::Format GetNriFormat(Format format)
{ return g_NrdFormatToNri[(uint32_t)format]; }

static inline uint64_t CreateDescriptorKey(uint64_t texture, bool isStorage)
{
    uint64_t key = uint64_t(isStorage ? 1 : 0) << 63ull;
    key |= texture & ((1ull << 63ull) - 1);

    return key;
}

template<typename T, typename A> constexpr T GetAlignedSize(const T& size, A alignment)
{
    return T(((size + alignment - 1) / alignment) * alignment);
}

bool Integration::Initialize(const IntegrationCreationDesc& integrationDesc, const InstanceCreationDesc& instanceDesc, nri::Device& device, const nri::CoreInterface& iCore)
{
    NRD_INTEGRATION_ASSERT(!m_Instance, "Already initialized! Did you forget to call 'Destroy'?");
    NRD_INTEGRATION_ASSERT(!integrationDesc.promoteFloat16to32 || !integrationDesc.demoteFloat32to16, "Can't be 'true' for both");
    NRD_INTEGRATION_ASSERT(integrationDesc.queuedFrameNum, "Can't be 0");

    const nri::DeviceDesc& deviceDesc = iCore.GetDeviceDesc(device);
    if (deviceDesc.nriVersionMajor != NRI_VERSION_MAJOR || deviceDesc.nriVersionMinor != NRI_VERSION_MINOR)
    {
        NRD_INTEGRATION_ASSERT(false, "NRI version mismatch detected!");
        return false;
    }

    const LibraryDesc& libraryDesc = GetLibraryDesc();
    if (libraryDesc.versionMajor != NRD_VERSION_MAJOR || libraryDesc.versionMinor != NRD_VERSION_MINOR)
    {
        NRD_INTEGRATION_ASSERT(false, "NRD version mismatch detected!");
        return false;
    }

    if (nri::nriGetInterface(device, NRI_INTERFACE(nri::HelperInterface), &m_iHelper) != nri::Result::SUCCESS)
        return false;

    if (CreateInstance(instanceDesc, m_Instance) != Result::SUCCESS)
        return false;

    m_QueuedFrameNum = integrationDesc.queuedFrameNum;
    m_EnableDescriptorCaching = integrationDesc.enableDescriptorCaching;
    m_PromoteFloat16to32 = integrationDesc.promoteFloat16to32;
    m_DemoteFloat32to16 = integrationDesc.demoteFloat32to16;
    m_Device = &device;
    m_iCore = &iCore;

    strncpy(m_Name, integrationDesc.name, sizeof(m_Name));

#if( NRD_INTEGRATION_DEBUG_LOGGING == 1 )
    char filename[128];
    snprintf(filename, sizeof(filename), "NRD-%s.log", m_Name);
    m_Log = fopen(filename, "w");
    if (m_Log)
        fprintf(m_Log, "Resource size = %u x %u\n", integrationDesc.resourceWidth, integrationDesc.resourceHeight);
#endif

    CreatePipelines();
    CreateResources(integrationDesc.resourceWidth, integrationDesc.resourceHeight);

    return true;
}

void Integration::CreatePipelines()
{
    // Assuming that the device is in IDLE state
    for (nri::Pipeline* pipeline : m_Pipelines)
        m_iCore->DestroyPipeline(*pipeline);
    m_Pipelines.clear();

#ifdef PROJECT_NAME
     utils::ShaderCodeStorage shaderCodeStorage;
#endif

    const InstanceDesc& instanceDesc = GetInstanceDesc(*m_Instance);
    const nri::DeviceDesc& deviceDesc = m_iCore->GetDeviceDesc(*m_Device);

    uint32_t constantBufferOffset = 0;
    uint32_t samplerOffset = 0;
    uint32_t textureOffset = 0;
    uint32_t storageTextureAndBufferOffset = 0;
    if (m_iCore->GetDeviceDesc(*m_Device).graphicsAPI == nri::GraphicsAPI::VK)
    {
        const LibraryDesc& nrdLibraryDesc = GetLibraryDesc();
        constantBufferOffset = nrdLibraryDesc.spirvBindingOffsets.constantBufferOffset;
        samplerOffset = nrdLibraryDesc.spirvBindingOffsets.samplerOffset;
        textureOffset = nrdLibraryDesc.spirvBindingOffsets.textureOffset;
        storageTextureAndBufferOffset = nrdLibraryDesc.spirvBindingOffsets.storageTextureAndBufferOffset;
    }

    // Allocate memory for descriptor sets
    uint32_t descriptorSetSamplersIndex = instanceDesc.constantBufferSpaceIndex == instanceDesc.samplersSpaceIndex ? 0 : 1;
    uint32_t descriptorSetResourcesIndex = instanceDesc.resourcesSpaceIndex == instanceDesc.constantBufferSpaceIndex ? 0 : (instanceDesc.resourcesSpaceIndex == instanceDesc.samplersSpaceIndex ? descriptorSetSamplersIndex : descriptorSetSamplersIndex + 1);
    uint32_t descriptorSetNum = std::max(descriptorSetSamplersIndex, descriptorSetResourcesIndex) + 1;

    nri::DescriptorSetDesc* descriptorSetDescs = (nri::DescriptorSetDesc*)alloca(sizeof(nri::DescriptorSetDesc) * descriptorSetNum);
    memset(descriptorSetDescs, 0, sizeof(nri::DescriptorSetDesc) * descriptorSetNum);

    nri::DescriptorSetDesc& descriptorSetConstantBuffer = descriptorSetDescs[0];
    descriptorSetConstantBuffer.registerSpace = instanceDesc.constantBufferSpaceIndex;

    nri::DescriptorSetDesc& descriptorSetSamplers = descriptorSetDescs[descriptorSetSamplersIndex];
    descriptorSetSamplers.registerSpace = instanceDesc.samplersSpaceIndex;

    nri::DescriptorSetDesc& descriptorSetResources = descriptorSetDescs[descriptorSetResourcesIndex];
    descriptorSetResources.registerSpace = instanceDesc.resourcesSpaceIndex;

    // Allocate memory for descriptor ranges
    uint32_t resourceRangesNum = 0;
    for (uint32_t i = 0; i < instanceDesc.pipelinesNum; i++)
    {
        const PipelineDesc& nrdPipelineDesc = instanceDesc.pipelines[i];
        resourceRangesNum = std::max(resourceRangesNum, nrdPipelineDesc.resourceRangesNum);
    }
    resourceRangesNum += 1; // samplers

    nri::DescriptorRangeDesc* descriptorRanges = (nri::DescriptorRangeDesc*)alloca(sizeof(nri::DescriptorRangeDesc) * resourceRangesNum);
    memset(descriptorRanges, 0, sizeof(nri::DescriptorRangeDesc) * resourceRangesNum);

    nri::DescriptorRangeDesc* samplersRange = descriptorRanges;
    nri::DescriptorRangeDesc* resourcesRanges = descriptorRanges + 1;

    // Constant buffer
    const nri::DynamicConstantBufferDesc dynamicConstantBufferDesc = {constantBufferOffset + instanceDesc.constantBufferRegisterIndex, nri::StageBits::COMPUTE_SHADER};
    descriptorSetConstantBuffer.dynamicConstantBuffers = &dynamicConstantBufferDesc;

    // Samplers
    samplersRange->descriptorType = nri::DescriptorType::SAMPLER;
    samplersRange->baseRegisterIndex = samplerOffset + instanceDesc.samplersBaseRegisterIndex;
    samplersRange->descriptorNum = instanceDesc.samplersNum;
    samplersRange->shaderStages =  nri::StageBits::COMPUTE_SHADER;

    // Pipelines
    for (uint32_t i = 0; i < instanceDesc.pipelinesNum; i++)
    {
        const PipelineDesc& nrdPipelineDesc = instanceDesc.pipelines[i];
        const ComputeShaderDesc& nrdComputeShader = (&nrdPipelineDesc.computeShaderDXBC)[std::max((int32_t)deviceDesc.graphicsAPI - 1, 0)];

        // Resources
        for (uint32_t j = 0; j < nrdPipelineDesc.resourceRangesNum; j++)
        {
            const ResourceRangeDesc& nrdResourceRange = nrdPipelineDesc.resourceRanges[j];

            if (nrdResourceRange.descriptorType == DescriptorType::TEXTURE)
            {
                resourcesRanges[j].baseRegisterIndex = textureOffset + nrdResourceRange.baseRegisterIndex;
                resourcesRanges[j].descriptorType = nri::DescriptorType::TEXTURE;
            }
            else
            {
                resourcesRanges[j].baseRegisterIndex = storageTextureAndBufferOffset + nrdResourceRange.baseRegisterIndex;
                resourcesRanges[j].descriptorType = nri::DescriptorType::STORAGE_TEXTURE;
            }

            resourcesRanges[j].descriptorNum = nrdResourceRange.descriptorsNum;
            resourcesRanges[j].shaderStages = nri::StageBits::COMPUTE_SHADER;
        }

        // Descriptor sets
        if (instanceDesc.resourcesSpaceIndex != instanceDesc.samplersSpaceIndex)
        {
            descriptorSetSamplers.rangeNum = 1;
            descriptorSetSamplers.ranges = samplersRange;

            descriptorSetResources.ranges = resourcesRanges;
            descriptorSetResources.rangeNum = nrdPipelineDesc.resourceRangesNum;
        }
        else
        {
            descriptorSetResources.ranges = descriptorRanges;
            descriptorSetResources.rangeNum = nrdPipelineDesc.resourceRangesNum + 1;
        }

        descriptorSetConstantBuffer.dynamicConstantBufferNum = nrdPipelineDesc.hasConstantData ? 1 : 0;

        // Pipeline layout
        nri::PipelineLayoutDesc pipelineLayoutDesc = {};
        pipelineLayoutDesc.descriptorSetNum = descriptorSetNum;
        pipelineLayoutDesc.descriptorSets = descriptorSetDescs;
        pipelineLayoutDesc.shaderStages = nri::StageBits::COMPUTE_SHADER;
        pipelineLayoutDesc.flags = nri::PipelineLayoutBits::IGNORE_GLOBAL_SPIRV_OFFSETS;

        nri::PipelineLayout* pipelineLayout = nullptr;
        NRD_INTEGRATION_ABORT_ON_FAILURE(m_iCore->CreatePipelineLayout(*m_Device, pipelineLayoutDesc, pipelineLayout));
        m_PipelineLayouts.push_back(pipelineLayout);

        // Pipeline
        nri::ShaderDesc computeShader = {};
    #ifdef PROJECT_NAME
        if (nrdComputeShader.bytecode && !m_ReloadShaders)
        {
    #endif
            computeShader.bytecode = nrdComputeShader.bytecode;
            computeShader.size = nrdComputeShader.size;
            computeShader.entryPointName = nrdPipelineDesc.shaderEntryPointName;
            computeShader.stage = nri::StageBits::COMPUTE_SHADER;
    #ifdef PROJECT_NAME
        }
        else
            computeShader = utils::LoadShader(deviceDesc.graphicsAPI, nrdPipelineDesc.shaderFileName, shaderCodeStorage, nrdPipelineDesc.shaderEntryPointName);
    #endif

        nri::ComputePipelineDesc pipelineDesc = {};
        pipelineDesc.pipelineLayout = pipelineLayout;
        pipelineDesc.shader = computeShader;

        nri::Pipeline* pipeline = nullptr;
        NRD_INTEGRATION_ABORT_ON_FAILURE(m_iCore->CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    m_ReloadShaders = true;
}

void Integration::CreateResources(uint16_t resourceWidth, uint16_t resourceHeight)
{
    const InstanceDesc& instanceDesc = GetInstanceDesc(*m_Instance);
    const uint32_t poolSize = instanceDesc.permanentPoolSize + instanceDesc.transientPoolSize;

    m_TexturePool.resize(poolSize); // No reallocation!

    // Texture pool
    for (uint32_t i = 0; i < poolSize; i++)
    {
        // Create NRI texture
        const TextureDesc& nrdTextureDesc = (i < instanceDesc.permanentPoolSize) ? instanceDesc.permanentPool[i] : instanceDesc.transientPool[i - instanceDesc.permanentPoolSize];

        nri::Format format = GetNriFormat(nrdTextureDesc.format);
        if (m_PromoteFloat16to32)
        {
            if (format == nri::Format::R16_SFLOAT)
                format = nri::Format::R32_SFLOAT;
            else if (format == nri::Format::RG16_SFLOAT)
                format = nri::Format::RG32_SFLOAT;
            else if (format == nri::Format::RGBA16_SFLOAT)
                format = nri::Format::RGBA32_SFLOAT;
        }
        else if (m_DemoteFloat32to16)
        {
            if (format == nri::Format::R32_SFLOAT)
                format = nri::Format::R16_SFLOAT;
            else if (format == nri::Format::RG32_SFLOAT)
                format = nri::Format::RG16_SFLOAT;
            else if (format == nri::Format::RGBA32_SFLOAT)
                format = nri::Format::RGBA16_SFLOAT;
        }

        uint16_t w = DivideUp(resourceWidth, nrdTextureDesc.downsampleFactor);
        uint16_t h = DivideUp(resourceHeight, nrdTextureDesc.downsampleFactor);

        nri::TextureDesc textureDesc = {};
        textureDesc.type = nri::TextureType::TEXTURE_2D;
        textureDesc.usage = nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE;
        textureDesc.format = format;
        textureDesc.width = w;
        textureDesc.height = h;
        textureDesc.mipNum = 1;

        nri::Texture* texture = nullptr;
        NRD_INTEGRATION_ABORT_ON_FAILURE(m_iCore->CreateTexture(*m_Device, textureDesc, texture));

        char name[128];
        if (i < instanceDesc.permanentPoolSize)
            snprintf(name, sizeof(name), "%s::P(%u)", m_Name, i);
        else
            snprintf(name, sizeof(name), "%s::T(%u)", m_Name, i - instanceDesc.permanentPoolSize);
        m_iCore->SetDebugName(texture, name);

        // Construct NRD texture
        nri::TextureBarrierDesc& nrdTexture = m_TexturePool[i];
        nrdTexture = nri::TextureBarrierFromUnknown(texture, {nri::AccessBits::UNKNOWN, nri::Layout::UNKNOWN}, 0, 1);

        // Adjust memory usage
        nri::MemoryDesc memoryDesc = {};
        m_iCore->GetTextureMemoryDesc(*texture, nri::MemoryLocation::DEVICE, memoryDesc);

        if (i < instanceDesc.permanentPoolSize)
            m_PermanentPoolSize += memoryDesc.size;
        else
            m_TransientPoolSize += memoryDesc.size;

    #if( NRD_INTEGRATION_DEBUG_LOGGING == 1 )
        if (m_Log)
            fprintf(m_Log, "%s\n\tformat=%u downsampleFactor=%u\n", name, nrdTextureDesc.format, nrdTextureDesc.downsampleFactor);
    #endif
    }

#if( NRD_INTEGRATION_DEBUG_LOGGING == 1 )
    if (m_Log)
        fprintf(m_Log, "%.1f Mb (permanent), %.1f Mb (transient)\n\n", double(m_PermanentPoolSize) / (1024.0f * 1024.0f), double(m_TransientPoolSize) / (1024.0f * 1024.0f));
#endif

    // Samplers
    for (uint32_t i = 0; i < instanceDesc.samplersNum; i++)
    {
        Sampler nrdSampler = instanceDesc.samplers[i];

        nri::SamplerDesc samplerDesc = {};
        samplerDesc.addressModes = {nri::AddressMode::CLAMP_TO_EDGE, nri::AddressMode::CLAMP_TO_EDGE};
        samplerDesc.filters.min = nrdSampler == Sampler::NEAREST_CLAMP ? nri::Filter::NEAREST : nri::Filter::LINEAR;
        samplerDesc.filters.mag = nrdSampler == Sampler::NEAREST_CLAMP ? nri::Filter::NEAREST : nri::Filter::LINEAR;

        nri::Descriptor* descriptor = nullptr;
        NRD_INTEGRATION_ABORT_ON_FAILURE(m_iCore->CreateSampler(*m_Device, samplerDesc, descriptor));
        m_Samplers.push_back(descriptor);
    }

    // Constant buffer
    const nri::DeviceDesc& deviceDesc = m_iCore->GetDeviceDesc(*m_Device);
    m_ConstantBufferViewSize = GetAlignedSize(instanceDesc.constantBufferMaxDataSize, deviceDesc.memoryAlignment.constantBufferOffset);
    m_ConstantBufferSize = uint64_t(m_ConstantBufferViewSize) * instanceDesc.descriptorPoolDesc.setsMaxNum * m_QueuedFrameNum;

    nri::BufferDesc bufferDesc = {};
    bufferDesc.size = m_ConstantBufferSize;
    bufferDesc.usage = nri::BufferUsageBits::CONSTANT_BUFFER;
    NRD_INTEGRATION_ABORT_ON_FAILURE(m_iCore->CreateBuffer(*m_Device, bufferDesc, m_ConstantBuffer));

    AllocateAndBindMemory();

    nri::BufferViewDesc constantBufferViewDesc = {};
    constantBufferViewDesc.viewType = nri::BufferViewType::CONSTANT;
    constantBufferViewDesc.buffer = m_ConstantBuffer;
    constantBufferViewDesc.size = m_ConstantBufferViewSize;
    NRD_INTEGRATION_ABORT_ON_FAILURE(m_iCore->CreateBufferView(constantBufferViewDesc, m_ConstantBufferView));

    // Descriptor pools
    nri::DescriptorPoolDesc descriptorPoolDesc = {};
    descriptorPoolDesc.descriptorSetMaxNum = instanceDesc.descriptorPoolDesc.setsMaxNum;
    descriptorPoolDesc.storageTextureMaxNum = instanceDesc.descriptorPoolDesc.storageTexturesMaxNum;
    descriptorPoolDesc.textureMaxNum = instanceDesc.descriptorPoolDesc.texturesMaxNum;
    descriptorPoolDesc.dynamicConstantBufferMaxNum = instanceDesc.descriptorPoolDesc.constantBuffersMaxNum;
    descriptorPoolDesc.samplerMaxNum = instanceDesc.descriptorPoolDesc.samplersMaxNum;

    for (uint32_t i = 0; i < m_QueuedFrameNum; i++)
    {
        nri::DescriptorPool* descriptorPool = nullptr;
        NRD_INTEGRATION_ABORT_ON_FAILURE(m_iCore->CreateDescriptorPool(*m_Device, descriptorPoolDesc, descriptorPool));
        m_DescriptorPools.push_back(descriptorPool);

        m_DescriptorSetSamplers.push_back(nullptr);
        m_DescriptorsInFlight.push_back({});
    }

    m_Width = resourceWidth;
    m_Height = resourceHeight;

#if( NRD_INTEGRATION_DEBUG_LOGGING == 1 )
    if (m_Log)
        fflush(m_Log);
#endif
}

void Integration::AllocateAndBindMemory()
{
    std::vector<nri::Texture*> textures(m_TexturePool.size(), nullptr);
    for (size_t i = 0; i < m_TexturePool.size(); i++)
        textures[i] = m_TexturePool[i].texture;

    nri::ResourceGroupDesc resourceGroupDesc = {};
    resourceGroupDesc.memoryLocation = nri::MemoryLocation::DEVICE;
    resourceGroupDesc.textureNum = (uint32_t)textures.size();
    resourceGroupDesc.textures = textures.data();

    size_t baseAllocation = m_MemoryAllocations.size();
    const size_t allocationNum = m_iHelper.CalculateAllocationNumber(*m_Device, resourceGroupDesc);
    m_MemoryAllocations.resize(baseAllocation + allocationNum, nullptr);
    NRD_INTEGRATION_ABORT_ON_FAILURE(m_iHelper.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data() + baseAllocation));

    resourceGroupDesc = {};
    resourceGroupDesc.memoryLocation = nri::MemoryLocation::HOST_UPLOAD;
    resourceGroupDesc.bufferNum = 1;
    resourceGroupDesc.buffers = &m_ConstantBuffer;

    baseAllocation = m_MemoryAllocations.size();
    m_MemoryAllocations.resize(baseAllocation + 1, nullptr);
    NRD_INTEGRATION_ABORT_ON_FAILURE(m_iHelper.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data() + baseAllocation));
}

void Integration::NewFrame()
{
    NRD_INTEGRATION_ASSERT(m_Instance, "Uninitialized! Did you forget to call 'Initialize'?");

    // Must be here since the initial value is "-1", otherwise "descriptorPool[0]" will be used twice on the 1st and 2nd frames
    m_FrameIndex++;

#if( NRD_INTEGRATION_DEBUG_LOGGING == 1 )
    if (m_Log)
    {
        fflush(m_Log);
        fprintf(m_Log, "(frame %u) ==============================================================================\n\n", m_FrameIndex);
    }
#endif

    // Current descriptor pool index
    m_DescriptorPoolIndex = m_FrameIndex % m_QueuedFrameNum;

    // Reset descriptor pool and samplers (since they are allocated from it)
    nri::DescriptorPool* descriptorPool = m_DescriptorPools[m_DescriptorPoolIndex];
    m_iCore->ResetDescriptorPool(*descriptorPool);

    m_DescriptorSetSamplers[m_DescriptorPoolIndex] = nullptr;

    // Referenced by the GPU descriptors can't be destroyed...
    if (!m_EnableDescriptorCaching)
    {
        for (const auto& entry : m_DescriptorsInFlight[m_DescriptorPoolIndex])
            m_iCore->DestroyDescriptor(*entry);
        m_DescriptorsInFlight[m_DescriptorPoolIndex].clear();
    }

    m_PrevFrameIndexFromSettings++;
}

bool Integration::SetCommonSettings(const CommonSettings& commonSettings)
{
    NRD_INTEGRATION_ASSERT(m_Instance, "Uninitialized! Did you forget to call 'Initialize'?");
    NRD_INTEGRATION_ASSERT(commonSettings.resourceSize[0] == commonSettings.resourceSizePrev[0]
        && commonSettings.resourceSize[1] == commonSettings.resourceSizePrev[1]
        && commonSettings.resourceSize[0] == m_Width && commonSettings.resourceSize[1] == m_Height,
        "NRD integration preallocates resources statically: DRS is only supported via 'rectSize / rectSizePrev'");

    Result result = nrd::SetCommonSettings(*m_Instance, commonSettings);
    NRD_INTEGRATION_ASSERT(result == Result::SUCCESS, "SetCommonSettings(): failed!");

    if (m_FrameIndex == 0 || commonSettings.accumulationMode != AccumulationMode::CONTINUE)
        m_PrevFrameIndexFromSettings = commonSettings.frameIndex;
    else
        NRD_INTEGRATION_ASSERT(m_PrevFrameIndexFromSettings == commonSettings.frameIndex, "'frameIndex' must be incremented by 1 on each frame");

    return result == Result::SUCCESS;
}

bool Integration::SetDenoiserSettings(Identifier denoiser, const void* denoiserSettings)
{
    NRD_INTEGRATION_ASSERT(m_Instance, "Uninitialized! Did you forget to call 'Initialize'?");

    Result result = nrd::SetDenoiserSettings(*m_Instance, denoiser, denoiserSettings);
    NRD_INTEGRATION_ASSERT(result == Result::SUCCESS, "SetDenoiserSettings(): failed!");

    return result == Result::SUCCESS;
}

void Integration::Denoise(const Identifier* denoisers, uint32_t denoisersNum, nri::CommandBuffer& commandBuffer, UserPool& userPool, bool restoreInitialState)
{
    NRD_INTEGRATION_ASSERT(m_Instance, "Uninitialized! Did you forget to call 'Initialize'?");

    // Save initial state
    nri::TextureBarrierDesc* initialStates = (nri::TextureBarrierDesc*)alloca(sizeof(nri::TextureBarrierDesc) * userPool.size());
    if (restoreInitialState)
    {
        for (size_t i = 0; i < userPool.size(); i++)
        {
            const nri::TextureBarrierDesc* nrdTexture = userPool[i];
            if (nrdTexture)
                initialStates[i] = *nrdTexture;
        }
    }

    // One time sanity check
    if (m_FrameIndex == 0)
    {
        const nri::Texture* normalRoughnessTexture = userPool[(size_t)ResourceType::IN_NORMAL_ROUGHNESS]->texture;
        const nri::TextureDesc& normalRoughnessDesc = m_iCore->GetTextureDesc(*normalRoughnessTexture);
        const LibraryDesc& nrdLibraryDesc = GetLibraryDesc();

        bool isNormalRoughnessFormatValid = false;
        switch(nrdLibraryDesc.normalEncoding)
        {
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
    if (!m_EnableDescriptorCaching)
        m_CachedDescriptors.clear();

    // Set descriptor pool
    nri::DescriptorPool* descriptorPool = m_DescriptorPools[m_DescriptorPoolIndex];
    m_iCore->CmdSetDescriptorPool(commandBuffer, *descriptorPool);

    // Invoke dispatches
    constexpr uint32_t lawnGreen = 0xFF7CFC00;
    constexpr uint32_t limeGreen = 0xFF32CD32;

    for (uint32_t i = 0; i < dispatchDescsNum; i++)
    {
        const DispatchDesc& dispatchDesc = dispatchDescs[i];
        m_iCore->CmdBeginAnnotation(commandBuffer, dispatchDesc.name, (i & 0x1) ? lawnGreen : limeGreen);

        Dispatch(commandBuffer, *descriptorPool, dispatchDesc, userPool);

        m_iCore->CmdEndAnnotation(commandBuffer);
    }

    // Restore state
    if (restoreInitialState)
    {
        nri::TextureBarrierDesc* uniqueBarriers = initialStates;
        uint32_t uniqueBarrierNum = 0;

        for (size_t i = 0; i < userPool.size(); i++)
        {
            nri::TextureBarrierDesc* nrdTexture = userPool[i];
            if (!nrdTexture)
                continue;

            const nri::TextureBarrierDesc* nrdTextureInitial = &initialStates[i];
            if (nrdTexture->after.access != nrdTextureInitial->after.access || nrdTexture->after.layout != nrdTextureInitial->after.layout)
            {
                nrdTexture->before = nrdTexture->after;
                nrdTexture->after = nrdTextureInitial->after;

                bool isDifferent = nrdTexture->after.access != nrdTexture->before.access || nrdTexture->after.layout != nrdTexture->before.layout;
                bool isUnknown = nrdTexture->after.access == nri::AccessBits::UNKNOWN || nrdTexture->after.layout == nri::Layout::UNKNOWN;
                if (isDifferent && !isUnknown)
                    uniqueBarriers[uniqueBarrierNum++] = *nrdTexture;
            }
        }

        if (uniqueBarrierNum)
        {
            nri::BarrierGroupDesc transitionBarriers = {};
            transitionBarriers.textures = uniqueBarriers;
            transitionBarriers.textureNum = uniqueBarrierNum;

            m_iCore->CmdBarrier(commandBuffer, transitionBarriers);
        }
    }
}

void Integration::Dispatch(nri::CommandBuffer& commandBuffer, nri::DescriptorPool& descriptorPool, const DispatchDesc& dispatchDesc, UserPool& userPool)
{
    const InstanceDesc& instanceDesc = GetInstanceDesc(*m_Instance);
    const PipelineDesc& pipelineDesc = instanceDesc.pipelines[dispatchDesc.pipelineIndex];

    nri::Descriptor** descriptors = (nri::Descriptor**)alloca(sizeof(nri::Descriptor*) * dispatchDesc.resourcesNum);
    memset(descriptors, 0, sizeof(nri::Descriptor*) * dispatchDesc.resourcesNum);

    nri::DescriptorRangeUpdateDesc* resourceRanges = (nri::DescriptorRangeUpdateDesc*)alloca(sizeof(nri::DescriptorRangeUpdateDesc) * pipelineDesc.resourceRangesNum);
    memset(resourceRanges, 0, sizeof(nri::DescriptorRangeUpdateDesc) * pipelineDesc.resourceRangesNum);

    nri::TextureBarrierDesc* transitions = (nri::TextureBarrierDesc*)alloca(sizeof(nri::TextureBarrierDesc) * dispatchDesc.resourcesNum);
    memset(transitions, 0, sizeof(nri::TextureBarrierDesc) * dispatchDesc.resourcesNum);

    nri::BarrierGroupDesc transitionBarriers = {};
    transitionBarriers.textures = transitions;

    uint32_t n = 0;
    for (uint32_t i = 0; i < pipelineDesc.resourceRangesNum; i++)
    {
        const ResourceRangeDesc& resourceRange = pipelineDesc.resourceRanges[i];
        const bool isStorage = resourceRange.descriptorType == DescriptorType::STORAGE_TEXTURE;

        resourceRanges[i].descriptors = descriptors + n;
        resourceRanges[i].descriptorNum = resourceRange.descriptorsNum;

        for (uint32_t j = 0; j < resourceRange.descriptorsNum; j++)
        {
            const ResourceDesc& nrdResource = dispatchDesc.resources[n];

            // Get texture
            nri::TextureBarrierDesc* nrdTexture = nullptr;
            if (nrdResource.type == ResourceType::TRANSIENT_POOL)
                nrdTexture = &m_TexturePool[nrdResource.indexInPool + instanceDesc.permanentPoolSize];
            else if (nrdResource.type == ResourceType::PERMANENT_POOL)
                nrdTexture = &m_TexturePool[nrdResource.indexInPool];
            else
            {
                nrdTexture = userPool[(uint32_t)nrdResource.type];
                NRD_INTEGRATION_ASSERT(nrdTexture && nrdTexture->texture, "'userPool' entry can't be NULL if it's in use!");
            }

            // Prepare barrier
            nri::AccessLayoutStage next = {};
            if (nrdResource.descriptorType == DescriptorType::TEXTURE)
                next = {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE, nri::StageBits::COMPUTE_SHADER};
            else
                next = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER};

            bool isStateChanged = next.access != nrdTexture->after.access || next.layout != nrdTexture->after.layout;
            bool isStorageBarrier = next.access == nri::AccessBits::SHADER_RESOURCE_STORAGE && nrdTexture->after.access == nri::AccessBits::SHADER_RESOURCE_STORAGE;
            if (isStateChanged || isStorageBarrier)
                transitions[transitionBarriers.textureNum++] = nri::TextureBarrierFromState(*nrdTexture, next);

            // Create descriptor
            uint64_t resource = m_iCore->GetTextureNativeObject(*nrdTexture->texture);
            uint64_t key = CreateDescriptorKey(resource, isStorage);
            const auto& entry = m_CachedDescriptors.find(key);

            nri::Descriptor* descriptor = nullptr;
            if (entry == m_CachedDescriptors.end())
            {
                const nri::TextureDesc& textureDesc = m_iCore->GetTextureDesc(*nrdTexture->texture);

                nri::Texture2DViewDesc desc = {nrdTexture->texture, isStorage ? nri::Texture2DViewType::SHADER_RESOURCE_STORAGE_2D : nri::Texture2DViewType::SHADER_RESOURCE_2D, textureDesc.format, 0, 1};
                NRD_INTEGRATION_ABORT_ON_FAILURE(m_iCore->CreateTexture2DView(desc, descriptor));

                m_CachedDescriptors.insert( std::make_pair(key, descriptor) );
                m_DescriptorsInFlight[m_DescriptorPoolIndex].push_back(descriptor);
            }
            else
                descriptor = entry->second;

            // Add descriptor to the range
            descriptors[n++] = descriptor;
        }
    }

    // Barriers
    m_iCore->CmdBarrier(commandBuffer, transitionBarriers);

    // Allocating descriptor sets
    uint32_t descriptorSetSamplersIndex = instanceDesc.constantBufferSpaceIndex == instanceDesc.samplersSpaceIndex ? 0 : 1;
    uint32_t descriptorSetResourcesIndex = instanceDesc.resourcesSpaceIndex == instanceDesc.constantBufferSpaceIndex ? 0 : (instanceDesc.resourcesSpaceIndex == instanceDesc.samplersSpaceIndex ? descriptorSetSamplersIndex : descriptorSetSamplersIndex + 1);
    uint32_t descriptorSetNum = std::max(descriptorSetSamplersIndex, descriptorSetResourcesIndex) + 1;
    bool samplersAreInSeparateSet = instanceDesc.samplersSpaceIndex != instanceDesc.constantBufferSpaceIndex && instanceDesc.samplersSpaceIndex != instanceDesc.resourcesSpaceIndex;

    nri::DescriptorSet** descriptorSets = (nri::DescriptorSet**)alloca(sizeof(nri::DescriptorSet*) * descriptorSetNum);
    nri::PipelineLayout* pipelineLayout = m_PipelineLayouts[dispatchDesc.pipelineIndex];

    for (uint32_t i = 0; i < descriptorSetNum; i++)
    {
        if (!samplersAreInSeparateSet || i != descriptorSetSamplersIndex)
            NRD_INTEGRATION_ABORT_ON_FAILURE(m_iCore->AllocateDescriptorSets(descriptorPool, *pipelineLayout, i, &descriptorSets[i], 1, 0));
    }

    // Updating constants
    uint32_t dynamicConstantBufferOffset = m_ConstantBufferOffsetPrev;
    if (dispatchDesc.constantBufferDataSize)
    {
        if (!dispatchDesc.constantBufferDataMatchesPreviousDispatch)
        {
            // Ring-buffer logic
            if (m_ConstantBufferOffset + m_ConstantBufferViewSize > m_ConstantBufferSize)
                m_ConstantBufferOffset = 0;

            // Upload CB data
            void* data = m_iCore->MapBuffer(*m_ConstantBuffer, m_ConstantBufferOffset, dispatchDesc.constantBufferDataSize);
            if (data)
            {
                memcpy(data, dispatchDesc.constantBufferData, dispatchDesc.constantBufferDataSize);
                m_iCore->UnmapBuffer(*m_ConstantBuffer);
            }

            // Ring-buffer logic
            dynamicConstantBufferOffset = m_ConstantBufferOffset;
            m_ConstantBufferOffset += m_ConstantBufferViewSize;

            // Save previous offset for potential CB data reuse
            m_ConstantBufferOffsetPrev = dynamicConstantBufferOffset;
        }

        m_iCore->UpdateDynamicConstantBuffers(*descriptorSets[0], 0, 1, &m_ConstantBufferView);
    }

    // Updating samplers
    const nri::DescriptorRangeUpdateDesc samplersDescriptorRange = {m_Samplers.data(), instanceDesc.samplersNum, 0};
    if (samplersAreInSeparateSet)
    {
        nri::DescriptorSet*& descriptorSetSamplers = m_DescriptorSetSamplers[m_DescriptorPoolIndex];
        if (!descriptorSetSamplers)
        {
            NRD_INTEGRATION_ABORT_ON_FAILURE(m_iCore->AllocateDescriptorSets(descriptorPool, *pipelineLayout, descriptorSetSamplersIndex, &descriptorSetSamplers, 1, 0));
            m_iCore->UpdateDescriptorRanges(*descriptorSetSamplers, 0, 1, &samplersDescriptorRange);
        }

        descriptorSets[descriptorSetSamplersIndex] = descriptorSetSamplers;
    }
    else
        m_iCore->UpdateDescriptorRanges(*descriptorSets[descriptorSetSamplersIndex], 0, 1, &samplersDescriptorRange);

    // Updating resources
    m_iCore->UpdateDescriptorRanges(*descriptorSets[descriptorSetResourcesIndex], instanceDesc.samplersSpaceIndex == instanceDesc.resourcesSpaceIndex ? 1 : 0, pipelineDesc.resourceRangesNum, resourceRanges);

    // Rendering
    m_iCore->CmdSetPipelineLayout(commandBuffer, *pipelineLayout);

    nri::Pipeline* pipeline = m_Pipelines[dispatchDesc.pipelineIndex];
    m_iCore->CmdSetPipeline(commandBuffer, *pipeline);

    for (uint32_t i = 0; i < descriptorSetNum; i++)
        m_iCore->CmdSetDescriptorSet(commandBuffer, i, *descriptorSets[i], i == 0 ? &dynamicConstantBufferOffset : nullptr);

    m_iCore->CmdDispatch(commandBuffer, {dispatchDesc.gridWidth, dispatchDesc.gridHeight, 1});

    // Debug logging
#if( NRD_INTEGRATION_DEBUG_LOGGING == 1 )
    if (m_Log)
    {
        fprintf(m_Log, "%c Pipeline #%u : %s\n\t", dispatchDesc.constantBufferDataMatchesPreviousDispatch ? ' ' : '!', dispatchDesc.pipelineIndex, dispatchDesc.name);
        for( uint32_t i = 0; i < dispatchDesc.resourcesNum; i++ )
        {
            const ResourceDesc& r = dispatchDesc.resources[i];

            if( r.type == ResourceType::PERMANENT_POOL )
                fprintf(m_Log, "P(%u) ", r.indexInPool);
            else if( r.type == ResourceType::TRANSIENT_POOL )
                fprintf(m_Log, "T(%u) ", r.indexInPool);
            else
            {
                const char* s = GetResourceTypeString(r.type);
                fprintf(m_Log, "%s ", s);
            }
        }
        fprintf(m_Log, "\n\n");
    }
#endif
}

void Integration::Destroy()
{
    NRD_INTEGRATION_ASSERT(m_Instance, "Already destroyed! Did you forget to call 'Initialize'?");

    m_iCore->DestroyDescriptor(*m_ConstantBufferView);
    m_iCore->DestroyBuffer(*m_ConstantBuffer);

    for (auto& descriptors : m_DescriptorsInFlight)
    {
        for (const auto& entry : descriptors)
            m_iCore->DestroyDescriptor(*entry);
        descriptors.clear();
    }
    m_DescriptorsInFlight.clear();
    m_CachedDescriptors.clear();

    for (const nri::TextureBarrierDesc& nrdTexture : m_TexturePool)
        m_iCore->DestroyTexture(*nrdTexture.texture);
    m_TexturePool.clear();

    for (nri::Descriptor* descriptor : m_Samplers)
        m_iCore->DestroyDescriptor(*descriptor);
    m_Samplers.clear();

    for (nri::Pipeline* pipeline : m_Pipelines)
        m_iCore->DestroyPipeline(*pipeline);
    m_Pipelines.clear();

    for (nri::PipelineLayout* pipelineLayout : m_PipelineLayouts)
        m_iCore->DestroyPipelineLayout(*pipelineLayout);
    m_PipelineLayouts.clear();

    for (nri::Memory* memory : m_MemoryAllocations)
        m_iCore->FreeMemory(*memory);
    m_MemoryAllocations.clear();

    for (nri::DescriptorPool* descriptorPool : m_DescriptorPools)
        m_iCore->DestroyDescriptorPool(*descriptorPool);
    m_DescriptorPools.clear();
    m_DescriptorSetSamplers.clear();

    DestroyInstance(*m_Instance);

    m_iCore = nullptr;
    m_Device = nullptr;
    m_ConstantBuffer = nullptr;
    m_ConstantBufferView = nullptr;
    m_Instance = nullptr;
    m_PermanentPoolSize = 0;
    m_TransientPoolSize = 0;
    m_ConstantBufferSize = 0;
    m_ConstantBufferViewSize = 0;
    m_ConstantBufferOffset = 0;
    m_QueuedFrameNum = 0;
    m_DescriptorPoolIndex = 0;
    m_FrameIndex = 0;
    m_ReloadShaders = false;
    m_EnableDescriptorCaching = false;

#if( NRD_INTEGRATION_DEBUG_LOGGING == 1 )
    if (m_Log)
        fclose(m_Log);
#endif
}

}
