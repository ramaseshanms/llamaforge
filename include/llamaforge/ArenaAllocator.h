#pragma once

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <string>

namespace llamaforge {

class LlamaResourceExhaustedException : public std::runtime_error {
public:
    explicit LlamaResourceExhaustedException(const std::string& msg) 
        : std::runtime_error("LlamaResourceExhausted: " + msg) {}
};

/**
 * @brief Fixed-size Arena Allocator for purely deterministic memory mapping.
 *
 * Pre-allocates a massive host byte array on initialization and serves all
 * subsequent requests via a lock-free bump pointer.
 * Designed specifically to back `ggml_context` and transient forward-pass graph activations.
 */
class ArenaAllocator {
public:
    explicit ArenaAllocator(size_t total_bytes);
    ~ArenaAllocator() = default;

    // Non-copyable, non-movable once bound to hardware
    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;
    ArenaAllocator(ArenaAllocator&&) = delete;
    ArenaAllocator& operator=(ArenaAllocator&&) = delete;

    /// Allocates `bytes` aligned to `alignment` boundaries.
    /// @throws LlamaResourceExhaustedException if the arena is full.
    void* Allocate(size_t bytes, size_t alignment = 256);

    /// Resets the bump pointer to 0. Does not physically free memory to the OS.
    /// Safely called after a forward pass completes.
    void Reset();

    size_t GetCapacity() const { return capacity_; }
    size_t GetUsed() const { return used_; }
    size_t GetRemaining() const { return capacity_ - used_; }

private:
    std::vector<uint8_t> buffer_; // Represents the raw OS pages physically pinned
    size_t capacity_;
    size_t used_ = 0;
};

} // namespace llamaforge
