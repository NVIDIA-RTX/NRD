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
struct StdAllocator {
    typedef T value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef std::true_type propagate_on_container_move_assignment;
    typedef std::false_type is_always_equal;

    StdAllocator(const nrd::AllocationCallbacks& allocationCallbacks)
        : m_Interface(allocationCallbacks) {
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

    const nrd::AllocationCallbacks& GetInterface() const {
        return m_Interface;
    }

    template <typename U>
    using other = StdAllocator<U>;

private:
    nrd::AllocationCallbacks m_Interface = {};
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
inline T* Allocate(StdAllocator<uint8_t>& allocator, Args&&... args) {
    const auto& lowLevelAllocator = allocator.GetInterface();
    T* object = (T*)lowLevelAllocator.Allocate(lowLevelAllocator.userArg, sizeof(T), alignof(T));

    new (object) T(std::forward<Args>(args)...);
    return object;
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
using Vector = std::vector<T, StdAllocator<T>>;
