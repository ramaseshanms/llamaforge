#include "llamaforge/SessionState.h"

namespace llamaforge {

SessionState::SessionState(uint32_t session_id, size_t max_context_size)
    : session_id_(session_id), max_context_size_(max_context_size) {
    // Arbitrary default quota, e.g. enough for max_context_size assuming 16 tokens/block
    block_table_.max_quota = (max_context_size + 15) / 16;
}

SessionState::~SessionState() = default;

SessionState::SessionState(SessionState&& other) noexcept 
    : session_id_(other.session_id_),
      status_(other.status_.load()),
      token_history_(std::move(other.token_history_)),
      current_pos_(other.current_pos_),
      max_context_size_(other.max_context_size_),
      block_table_(other.block_table_) {
    // Note: mutex cannot be copied/moved, the new instance gets a fresh unlocked mutex.
}

SessionState& SessionState::operator=(SessionState&& other) noexcept {
    if (this != &other) {
        session_id_ = other.session_id_;
        status_.store(other.status_.load());
        token_history_ = std::move(other.token_history_);
        current_pos_ = other.current_pos_;
        max_context_size_ = other.max_context_size_;
        block_table_ = other.block_table_;
    }
    return *this;
}

void SessionState::AppendTokens(const std::vector<int32_t>& tokens) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    token_history_.insert(token_history_.end(), tokens.begin(), tokens.end());
}

const std::vector<int32_t>& SessionState::GetTokens() const {
    return token_history_;
}

void SessionState::AdvancePos(uint32_t n_tokens) {
    current_pos_ += n_tokens;
}

uint32_t SessionState::GetCurrentPos() const {
    return current_pos_;
}

void SessionState::Suspend() {
    status_.store(SessionStatus::PAUSED);
}

void SessionState::Resume() {
    status_.store(SessionStatus::IDLE);
}

void SessionState::Reset() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    token_history_.clear();
    current_pos_ = 0;
    status_.store(SessionStatus::IDLE);
}

void SessionState::AllocateNextBlock(PagedKVCache& global_cache) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (block_table_.logical_to_physical.size() >= block_table_.max_quota) {
        throw LlamaResourceExhaustedException("Session Quota Exceeded. Cannot allocate more KV blocks.");
    }
    
    BlockId new_block = global_cache.AllocateBlock();
    block_table_.logical_to_physical.push_back(new_block);
}

void SessionState::ReleaseContext(PagedKVCache& global_cache) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    for (BlockId id : block_table_.logical_to_physical) {
        global_cache.FreeBlock(id);
    }
    block_table_.logical_to_physical.clear();
}

} // namespace llamaforge
