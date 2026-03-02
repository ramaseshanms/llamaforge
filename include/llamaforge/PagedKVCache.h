#pragma once

#include "ArenaAllocator.h"
#include <vector>
#include <cstdint>
#include <mutex>

namespace llamaforge {

using BlockId = uint32_t;
constexpr BlockId INVALID_BLOCK = static_cast<BlockId>(-1);

/**
 * @brief Represents the mapping of a single User Session's tokens to Physical Blocks.
 * Replaces linear memory in `SessionState` to allow infinite sliding-window contexts.
 */
struct VirtualBlockTable {
    std::vector<BlockId> logical_to_physical;
    size_t max_quota; // How many physical blocks this session is allowed at peak
};

/**
 * @brief Manages the global physical KV Cache Slabs to prevent cross-session memory starvation.
 * Contains free-lists and enforces session memory quotas deterministically.
 */
class PagedKVCache {
public:
    /// @param total_blocks The exact number of 2MB slabs reserved on hardware.
    explicit PagedKVCache(size_t total_blocks);
    ~PagedKVCache() = default;

    PagedKVCache(const PagedKVCache&) = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;

    /// Attempts to grab a free block for a session.
    /// @throws LlamaResourceExhaustedException if hardware is full and no eviction is possible.
    BlockId AllocateBlock();

    /// Releases a block back to the global pool immediately (O(1)).
    void FreeBlock(BlockId block_id);

    size_t GetTotalBlocks() const { return total_blocks_; }
    size_t GetFreeBlocks() const { return free_list_.size(); }

private:
    size_t total_blocks_;
    std::vector<BlockId> free_list_;
    std::mutex allocator_mutex_;
};

} // namespace llamaforge
