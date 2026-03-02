#pragma once

#include "SessionState.h"
#include "ModelStore.h"
#include "ArenaAllocator.h"
#include "PagedKVCache.h"

#include <memory>
#include <stdexcept>
#include <functional>

// Forward declarations for low-level abstractions
struct ggml_context;
struct llama_context;
struct ggml_tensor;

namespace llamaforge {

/**
 * @brief Represents the low-level execution environment for the inference graph.
 *
 * InferenceContext binds a loaded ModelStore and an active SessionState to perform
 * computation. It owns the scratch buffers (`ggml_context` representing computation tensors)
 * and thread-pool interactions (`llama_context`).
 *
 * Design constraints:
 * - Decoupled from SessionState meaning we can drop an Execution Context entirely (e.g., OOM),
 *   while the SessionState containing the logical conversation history remains intact.
 * - Extensibility: Inference steps (prefill, decode) emit callbacks so logic wrappers
 *   can intercept token outputs without touching the allocator/graph scheduler internals.
 */
class InferenceContext {
public:
    InferenceContext(std::shared_ptr<ModelStore> model, 
                     std::shared_ptr<PagedKVCache> global_kv,
                     size_t max_batch_size, 
                     size_t num_threads);
    ~InferenceContext();

    // Prevent moving or copying. Bound strictly to a worker thread's hardware resources.
    InferenceContext(const InferenceContext&) = delete;
    InferenceContext& operator=(const InferenceContext&) = delete;
    InferenceContext(InferenceContext&&) = delete;
    InferenceContext& operator=(InferenceContext&&) = delete;

    /// Attaches a session state for the next forward pass.
    void BindSession(std::shared_ptr<SessionState> session);
    
    /// Releases the active session context gracefully.
    void DetachSession();

    /// Evaluates prompt tokens (Prefill Phase).
    /// @param num_tokens The batch size to process in this step.
    /// @return Number of tokens successfully evaluated.
    size_t EvaluatePrompt(size_t num_tokens);

    /// Evaluates a single token (Decode Phase).
    /// @return The sampled next token.
    int32_t EvaluateNextToken();

    using TokenCallback = std::function<bool(int32_t token, const SessionState& state)>;
    
    /// Registers a callback triggered whenever a new token is generated.
    /// Returning 'false' from the callback aborts generation gracefully.
    void OnTokenGenerated(TokenCallback callback);

private:
    std::shared_ptr<ModelStore> model_;
    std::shared_ptr<PagedKVCache> global_kv_;
    std::shared_ptr<SessionState> current_session_;

    // Core ggml state for graph building 
    ggml_context* compute_ctx_ = nullptr;
    
    // Fixed hardware scratch pad (no dynamic mallocs during inference)
    std::unique_ptr<ArenaAllocator> scratch_arena_;

    // Core llama.cpp state for scheduled execution
    llama_context* llama_ctx_ = nullptr;

    size_t max_batch_size_;
    size_t num_threads_;
    
    TokenCallback token_cb_;

    // Internal allocator handles.
    // Encapsulates ggml_backend allocations so extending inference (e.g. adding speculative decoding)
    // only involves replacing the forward pass, not internal mmaps.
    void AllocateScratchBuffers();
    void ReleaseScratchBuffers();
};

} // namespace llamaforge
