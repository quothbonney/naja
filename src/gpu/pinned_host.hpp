#pragma once

#include "helper.h"

namespace naja::gpu {

struct PinnedHostBuffer {
    void* ptr = nullptr;
    size_t size = 0;

    PinnedHostBuffer() = default;

    explicit PinnedHostBuffer(size_t bytes, unsigned int flags = cudaHostAllocPortable)
        : size(bytes) {
        if (size > 0) {
            CUDA_CHECK(cudaHostAlloc(&ptr, size, flags));
        }
    }

    ~PinnedHostBuffer() {
        if (ptr) {
            cudaFreeHost(ptr);
            ptr = nullptr;
        }
    }

    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
        : ptr(other.ptr), size(other.size) {
        other.ptr = nullptr;
        other.size = 0;
    }

    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr) {
                cudaFreeHost(ptr);
            }
            ptr = other.ptr;
            size = other.size;
            other.ptr = nullptr;
            other.size = 0;
        }
        return *this;
    }

    void* data() { return ptr; }
    const void* data() const { return ptr; }

    template <typename T>
    T* as() {
        return static_cast<T*>(ptr);
    }

    template <typename T>
    const T* as() const {
        return static_cast<const T*>(ptr);
    }
};

} // namespace naja::gpu

