#include "llamaforge/InferenceContext.h"

namespace llamaforge {

InferenceContext::InferenceContext(std::shared_ptr<ModelStore> model, 
                                   std::shared_ptr<PagedKVCache> global_kv,
                                   size_t max_batch_size, 
                                   size_t num_threads)
    : model_(std::move(model)), global_kv_(std::move(global_kv)), max_batch_size_(max_batch_size), num_threads_(num_threads) {
    AllocateScratchBuffers();
}

InferenceContext::~InferenceContext() {
    ReleaseScratchBuffers();
}

void InferenceContext::BindSession(std::shared_ptr<SessionState> session) {
    current_session_ = std::move(session);
}

void InferenceContext::DetachSession() {
    current_session_.reset();
}

size_t InferenceContext::EvaluatePrompt(size_t /*num_tokens*/) {
    if (!current_session_) return 0;
    // Stub
    return 0;
}

int32_t InferenceContext::EvaluateNextToken() {
    if (!current_session_) return -1;
    // Stub
    return 0;
}

void InferenceContext::OnTokenGenerated(TokenCallback callback) {
    token_cb_ = std::move(callback);
}

void InferenceContext::AllocateScratchBuffers() {
    // e.g. 256MB fixed size for scratch evaluations
    scratch_arena_ = std::make_unique<ArenaAllocator>(256 * 1024 * 1024);
}

void InferenceContext::ReleaseScratchBuffers() {
    scratch_arena_.reset();
}

} // namespace llamaforge
