/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <array>
#include <vector>

template <typename T>
void StdAllocator_MaybeUnused([[maybe_unused]] const T& arg) {
}

#if _WIN32

#    include <malloc.h>

inline void* AlignedMalloc(void* userArg, size_t size, size_t alignment) {
    StdAllocator_MaybeUnused(userArg);

    return _aligned_malloc(size, alignment);
}

inline void* AlignedRealloc(void* userArg, void* memory, size_t size, size_t alignment) {
    StdAllocator_MaybeUnused(userArg);

    return _aligned_realloc(memory, size, alignment);
}

inline void AlignedFree(void* userArg, void* memory) {
    StdAllocator_MaybeUnused(userArg);

    _aligned_free(memory);
}

#elif defined(__linux__) || defined(__SCE__) || defined(__APPLE__)

#    include <cstdlib>

inline uint8_t* AlignMemory(uint8_t* memory, size_t alignment) {
    return (uint8_t*)((size_t(memory) + alignment - 1) & ~(alignment - 1));
}

inline void* AlignedMalloc(void* userArg, size_t size, size_t alignment) {
    StdAllocator_MaybeUnused(userArg);

    uint8_t* memory = (uint8_t*)malloc(size + sizeof(uint8_t*) + alignment - 1);

    if (memory == nullptr)
        return nullptr;

    uint8_t* alignedMemory = AlignMemory(memory + sizeof(uint8_t*), alignment);
    uint8_t** memoryHeader = (uint8_t**)alignedMemory - 1;
    *memoryHeader = memory;

    return alignedMemory;
}

inline void* AlignedRealloc(void* userArg, void* memory, size_t size, size_t alignment) {
    if (memory == nullptr)
        return AlignedMalloc(userArg, size, alignment);

    uint8_t** memoryHeader = (uint8_t**)memory - 1;
    uint8_t* oldMemory = *memoryHeader;
    uint8_t* newMemory = (uint8_t*)realloc(oldMemory, size + sizeof(uint8_t*) + alignment - 1);

    if (newMemory == nullptr)
        return nullptr;

    if (newMemory == oldMemory)
        return memory;

    uint8_t* alignedMemory = AlignMemory(newMemory + sizeof(uint8_t*), alignment);
    memoryHeader = (uint8_t**)alignedMemory - 1;
    *memoryHeader = newMemory;

    return alignedMemory;
}

inline void AlignedFree(void* userArg, void* memory) {
    StdAllocator_MaybeUnused(userArg);

    if (memory == nullptr)
        return;

    uint8_t** memoryHeader = (uint8_t**)memory - 1;
    uint8_t* oldMemory = *memoryHeader;
    free(oldMemory);
}

#endif

inline void CheckAndSetDefaultAllocator(AllocationCallbacks& allocationCallbacks) {
    if (allocationCallbacks.Allocate != nullptr)
        return;

    allocationCallbacks.Allocate = AlignedMalloc;
    allocationCallbacks.Reallocate = AlignedRealloc;
    allocationCallbacks.Free = AlignedFree;
}

template <typename T>
struct StdAllocator {
    typedef T value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef std::true_type propagate_on_container_move_assignment;
    typedef std::false_type is_always_equal;

    StdAllocator(const AllocationCallbacks& allocationCallbacks)
        : m_Interface(allocationCallbacks) {
        CheckAndSetDefaultAllocator(m_Interface);
    }

    StdAllocator(const StdAllocator<T>& allocator)
        : m_Interface(allocator.GetInterface()) {
    }

    template <class U>
    StdAllocator(const StdAllocator<U>& allocator)
        : m_Interface(allocator.GetInterface()) {
    }

    StdAllocator<T>& operator=(const StdAllocator<T>& allocator) {
        m_Interface = allocator.GetInterface();
        return *this;
    }

    T* allocate(size_t n) noexcept {
        return (T*)m_Interface.Allocate(m_Interface.userArg, n * sizeof(T), alignof(T));
    }

    void deallocate(T* memory, size_t) noexcept {
        m_Interface.Free(m_Interface.userArg, memory);
    }

    const AllocationCallbacks& GetInterface() const {
        return m_Interface;
    }

    template <typename U>
    using other = StdAllocator<U>;

private:
    AllocationCallbacks m_Interface = {};
};

template <typename T>
bool operator==(const StdAllocator<T>& left, const StdAllocator<T>& right) {
    return left.GetInterface() == right.GetInterface();
}

template <typename T>
bool operator!=(const StdAllocator<T>& left, const StdAllocator<T>& right) {
    return !operator==(left, right);
}

template <typename T>
inline T GetAlignedSize(const T& x, uint32_t alignment) {
    return ((x + alignment - 1) / alignment) * alignment;
}

template <typename T>
inline T* Align(T* x, size_t alignment) {
    return (T*)(((size_t)x + alignment - 1) / alignment * alignment);
}

template <typename T, uint32_t N>
constexpr uint32_t GetCountOf(T const (&)[N]) {
    return N;
}

template <typename T>
constexpr uint32_t GetCountOf(const std::vector<T>& v) {
    return (uint32_t)v.size();
}

template <typename T, uint32_t N>
constexpr uint32_t GetCountOf(const std::array<T, N>& v) {
    return (uint32_t)v.size();
}

template <typename T, typename... Args>
constexpr void Construct(T* objects, size_t number, Args&&... args) {
    for (size_t i = 0; i < number; i++)
        new (objects + i) T(std::forward<Args>(args)...);
}

template <typename T, typename... Args>
inline T* Allocate(StdAllocator<uint8_t>& allocator, Args&&... args) {
    const auto& lowLevelAllocator = allocator.GetInterface();
    T* object = (T*)lowLevelAllocator.Allocate(lowLevelAllocator.userArg, sizeof(T), alignof(T));

    new (object) T(std::forward<Args>(args)...);
    return object;
}

template <typename T, typename... Args>
inline T* AllocateArray(StdAllocator<uint8_t>& allocator, size_t arraySize, Args&&... args) {
    const auto& lowLevelAllocator = allocator.GetInterface();
    T* array = (T*)lowLevelAllocator.Allocate(lowLevelAllocator.userArg, arraySize * sizeof(T), alignof(T));

    for (size_t i = 0; i < arraySize; i++)
        new (array + i) T(std::forward<Args>(args)...);

    return array;
}

template <typename T>
inline void Deallocate(StdAllocator<uint8_t>& allocator, T* object) {
    if (object == nullptr)
        return;

    object->~T();

    const auto& lowLevelAllocator = allocator.GetInterface();
    lowLevelAllocator.Free(lowLevelAllocator.userArg, object);
}

template <typename T>
inline void DeallocateArray(StdAllocator<uint8_t>& allocator, T* array, size_t arraySize) {
    if (array == nullptr)
        return;

    for (size_t i = 0; i < arraySize; i++)
        (array + i)->~T();

    const auto& lowLevelAllocator = allocator.GetInterface();
    lowLevelAllocator.Free(lowLevelAllocator.userArg, array);
}

//==============================================================================================================================

template <typename T>
using Vector = std::vector<T, StdAllocator<T>>;
