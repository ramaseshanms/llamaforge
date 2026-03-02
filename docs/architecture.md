# LlamaForge Architecture & Core Abstractions

This document details the core runtime abstractions and ownership rules for the LlamaForge C++ runtime built around `llama.cpp` and `ggml`.

---

## 1. Core Abstractions

The system is decoupled into three distinct layers to provide production-grade fault tolerance, separation of concerns, and ease of extensibility.

### A. ModelStore (`ModelStore.h`)
**Responsibility**: Manages the lifetime of a loaded LLM (the weights, static metadata, and mmapped tensors).
- Outlives any specific inference session.
- Thread-safe and read-only post-initialization.
- **Ownership**: Owns the `llama_model` instance.

### B. SessionState (`SessionState.h`)
**Responsibility**: Represents the logical state of a generation request / conversation.
- Contains the KV-cache specific to one sequence, token history, and logical progress.
- Tied to the *user interaction*, completely decoupled from the execution hardware until evaluation.
- **Ownership**: Wraps `llama_kv_cache` (logical ownership of sequence slots).

### C. InferenceContext (`InferenceContext.h`)
**Responsibility**: The hardware-level execution environment for the compute graph.
- Binds a `ModelStore` and a `SessionState` to run the forward pass (prefill/decode).
- Reusable across different sessions over time.
- **Ownership**: Strictly owns the scratch buffers (`ggml_context` for the dynamic graph computations) and the `llama_context` representing the scheduled worker threads.

---

## 2. Ownership Rules & Memory Management

Production models demand strict tracking of large CPU/GPU allocations. LlamaForge implements the following ownership rules for underlying `ggml/llama.cpp` structs:

1. **`ggml_context`**: 
   - *Exclusively owned by `InferenceContext`.* The context used to build computational graphs is completely ephemeral and re-allocated/re-used *per forward pass* or *batch pass*. Never escapes the InferenceContext boundary.
2. **`ggml_tensor`**:
   - Static weight tensors are owned by the `ModelStore` (via `llama_model`).
   - Dynamically allocated graph node tensors exist purely within the `InferenceContext`'s `ggml_context` lifecycle. They are freed immediately after `llama_decode`.
3. **KV Cache (`llama_kv_cache` / KV buffer cells)**:
   - *Logically owned by `SessionState`.* The cache slots and logical sequence IDs belong to the session to permit swapping to disk/host RAM if the session is paused (`SessionState::Suspend()`).
   - The *physical allocation* of the KV block resides in the `llama_context` (managed by `InferenceContext`), but `SessionState` handles slot indexing and lifecycle.

---

## 3. Extensibility Design

A primary requirement for a production inference server is the ability to experiment with new generation techniques (e.g., speculative decoding, grammar-constrained sampling, prefix caching) *without* modifying low-level allocator or scheduler logic.

**How LlamaForge achieves extensibility:**
- **Callback Introspection**: `InferenceContext::OnTokenGenerated()` allows high-level wrappers to intercept the autoregressive decoding loop token-by-token.
- **Separation of Forward Pass vs Memory**: `InferenceContext::EvaluatePrompt()` handles graph execution internally. A new feature like *Prompt Caching* operates entirely within `SessionState` (copying KV blocks) before binding. `InferenceContext` does not need to know where the KV tokens originated.
- **Stateless Allocator Wrappers**: `AllocateScratchBuffers()` is private. By wrapping `ggml_backend` with a generic memory pool inside the InferenceContext, developers can plug in custom sampling (logits processing) without touching the tensor allocation math.
