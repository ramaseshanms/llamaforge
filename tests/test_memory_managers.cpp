#include "../include/llamaforge/ArenaAllocator.h"
#include "../include/llamaforge/PagedKVCache.h"
#include "../include/llamaforge/SessionState.h"
#include <iostream>
#include <cassert>

using namespace llamaforge;

void test_arena_allocator() {
    std::cout << "Running ArenaAllocator tests...\n";
    // 1MB Arena
    ArenaAllocator arena(1024 * 1024);
    assert(arena.GetCapacity() == 1024 * 1024);
    assert(arena.GetUsed() == 0);

    // Allocate 100 bytes aligned to 256
    void* ptr1 = arena.Allocate(100, 256);
    assert(ptr1 != nullptr);
    size_t ptr_val = reinterpret_cast<size_t>(ptr1);
    assert(ptr_val % 256 == 0);
    assert(arena.GetUsed() > 0);

    // Fill it up
    try {
        arena.Allocate(1024 * 1024, 256);
        assert(false && "Should have thrown out of memory");
    } catch (const LlamaResourceExhaustedException& e) {
        // Expected
    }

    // Reset
    arena.Reset();
    assert(arena.GetUsed() == 0);
    std::cout << "ArenaAllocator passed.\n";
}

void test_paged_kv_cache() {
    std::cout << "Running PagedKVCache tests...\n";
    // System with exactly 10 blocks
    PagedKVCache kv_cache(10);
    assert(kv_cache.GetTotalBlocks() == 10);
    assert(kv_cache.GetFreeBlocks() == 10);

    // Session with max context demanding a quota of 3 blocks
    SessionState session(1, /*max_context_size=*/ 3 * 16); 
    
    session.AllocateNextBlock(kv_cache);
    session.AllocateNextBlock(kv_cache);
    assert(session.GetBlockTable().logical_to_physical.size() == 2);
    assert(kv_cache.GetFreeBlocks() == 8);

    session.AllocateNextBlock(kv_cache);
    assert(session.GetBlockTable().logical_to_physical.size() == 3);
    
    // Attempting a 4th block should hit the Session Quota
    try {
        session.AllocateNextBlock(kv_cache);
        assert(false && "Should have thrown quota exceeded");
    } catch (const LlamaResourceExhaustedException& e) {
        // Expected
    }

    // Release context returning memory back to pool
    session.ReleaseContext(kv_cache);
    assert(session.GetBlockTable().logical_to_physical.empty());
    assert(kv_cache.GetFreeBlocks() == 10);
    
    std::cout << "PagedKVCache passed.\n";
}

int main() {
    try {
        test_arena_allocator();
        test_paged_kv_cache();
        std::cout << "All memory tests passed successfully!\n";
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
