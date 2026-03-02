#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <mutex>
#include <atomic>

// Forward declarations
struct llama_context;
struct llama_kv_cache;

#include "PagedKVCache.h"

namespace llamaforge {

class ModelStore;

enum class SessionStatus {
    IDLE,
    PREFILLING,
    DECODING,
    PAUSED,
    ERROR,
    TERMINATED
};

/**
 * @brief Represents the logical state of a generation request / conversation.
 *
 * SessionState owns the KV cache for a specific sequence, the token history,
 * and tracks the progress of the current query. SessionState is strictly tied to
 * the *user interaction*, not the execution machinery.
 *
 * Ownership:
 * - Owns the context/sequence state (llama_kv_cache wrapper)
 * - Owns logical token arrays (input prompt, decoded outputs)
 * - Contains locks, making it safe to transfer between executor threads.
 */
class SessionState {
public:
    SessionState(uint32_t session_id, size_t max_context_size);
    ~SessionState();

    // Move semantics allowed for session state transfer during recovery
    SessionState(SessionState&&) noexcept;
    SessionState& operator=(SessionState&&) noexcept;

    // Disallow copies (KV cache is tied uniquely to a session)
    SessionState(const SessionState&) = delete;
    SessionState& operator=(const SessionState&) = delete;

    /// Read/Write active prompt tokens
    void AppendTokens(const std::vector<int32_t>& tokens);
    const std::vector<int32_t>& GetTokens() const;

    /// Advances the internal sequence position counter after successful inference
    void AdvancePos(uint32_t n_tokens);
    uint32_t GetCurrentPos() const;

    /// Marks session as paused to free up execution slots (swaps KV to host ram if nested)
    void Suspend();
    void Resume();

    /// Clears the session logic for reuse
    void Reset();

    uint32_t GetId() const { return session_id_; }
    SessionStatus GetStatus() const { return status_.load(); }
    void SetStatus(SessionStatus s) { status_.store(s); }

    // KV Cache Block Paging
    /// Maps a new logical block to the next available physical block from the global cache.
    /// @throws LlamaResourceExhaustedException if quota exceeded or global OOM.
    void AllocateNextBlock(PagedKVCache& global_cache);
    
    /// Releases all held physical blocks back to the global cache (O(1) teardown).
    void ReleaseContext(PagedKVCache& global_cache);

    const VirtualBlockTable& GetBlockTable() const { return block_table_; }

private:
    uint32_t session_id_;
    std::atomic<SessionStatus> status_{SessionStatus::IDLE};
    
    std::vector<int32_t> token_history_;
    uint32_t current_pos_ = 0;
    size_t max_context_size_;

    VirtualBlockTable block_table_;
    std::mutex state_mutex_;
};

} // namespace llamaforge
