# Failure & Recovery Model

Production LLM inference must gracefully handle low-level memory issues and hardware pressure without process restarts. C++ runtime processes are difficult to crash-loop reliably, meaning stateful error handling is a primary requirement.

## 1. Enumerate Failure Modes

| Failure Mode | Detection Vector | Impact | Runtime Behavior |
|---|---|---|---|
| **OOM (Out Of Memory)** | `ggml_backend_alloc` returns `NULL`. | Cannot allocate KV blocks for new request. | Inference context drops request, session transitions to `ERROR`. Process remains alive. |
| **Corrupted KV Cache** | Sequence slot ID mismatch, failed checksum. | Context window reads wrong prefix. | Offending `SessionState` clears its logical KV history and enforces a full Prefill replay. |
| **Thread Starvation** | Pthread dispatch timeout / `llama_decode` > 1000ms budget. | Spike in tail latency (`p99`). | Timeout abort registered on `InferenceContext::OnTokenGenerated()`. Kills the current generation step early. |
| **Invalid Model Payload** | `llama_model_load` magic header mismatch. | App cannot serve requests. | Detected during Cold Start. `ModelStore::Load()` fails synchronously. Alerts upstream control plane instead of crashing. |

---

## 2. Deterministic Cleanup Paths

When a generation request fails mid-flight, LlamaForge guarantees memory safety natively in C++:

### Reclaiming Resources
Do not rely on OS reclaim (e.g. process death cleaning up shared memory).

1. **Abortion Signal (`TokenCallback` return `false`)**: 
   A hard interrupt from upper networking layers (e.g. gRPC client disconnect) triggers a `false` return inside `InferenceContext::OnTokenGenerated()`.
2. **Context Disassembly**: 
   The `InferenceContext` resets its internal `ggml_context` using `ggml_free`, returning buffer pages back to the runtime's memory pool instantly.
3. **Session Eviction**:
   `SessionState::Reset()` is called. The physical slots reserved in the `llama_kv_cache` buffer inside the bound `llama_context` are zeroed out (logically released).

---

## 3. Returning to Valid State (Without Process Restart)

LlamaForge separates user conversations (`SessionState`) from execution containers (`InferenceContext`). This makes recovery robust:

- **Handling a Failed Node**: 
  If an `InferenceContext` throws a low-level error (memory segmentation or unhandled struct layout), *the entire InferenceContext wrapper is destroyed*.
- **State Salvage**: 
  The `SessionState` itself remains uncorrupted. The failed `InferenceContext` is swapped with a fresh one from the runtime pool. 
- **Resumption**:
  The system binds the salvaged `SessionState` into the robust `InferenceContext` and attempts to synthesize the logical history. If necessary, it issues an immediate Prefill command against the prompt history (`GetTokens()`) to rehydrate a fresh hardware KV-Block, entirely transparent to the user.
