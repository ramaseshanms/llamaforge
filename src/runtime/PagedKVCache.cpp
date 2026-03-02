#include "llamaforge/PagedKVCache.h"
#include <stdexcept>

namespace llamaforge {

PagedKVCache::PagedKVCache(size_t total_blocks) : total_blocks_(total_blocks) {
    // Initialize the free list with all available physical blocks [0 .. N-1]
    free_list_.reserve(total_blocks);
    for (size_t i = 0; i < total_blocks; ++i) {
        // Build reverse so pop_back() gives 0 first.
        free_list_.push_back(static_cast<BlockId>(total_blocks - 1 - i));
    }
}

BlockId PagedKVCache::AllocateBlock() {
    std::lock_guard<std::mutex> lock(allocator_mutex_);
    
    if (free_list_.empty()) {
        throw LlamaResourceExhaustedException("PagedKVCache: OOM. No physical KV slabs remaining.");
    }

    BlockId allocated = free_list_.back();
    free_list_.pop_back();
    return allocated;
}

void PagedKVCache::FreeBlock(BlockId block_id) {
    if (block_id >= total_blocks_) {
        // Prevent corrupting the free list with invalid IDs
        throw std::out_of_range("Invalid BlockId passed to FreeBlock");
    }

    std::lock_guard<std::mutex> lock(allocator_mutex_);
    free_list_.push_back(block_id);
}

} // namespace llamaforge
