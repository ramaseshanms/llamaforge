# LlamaForge Startup & Initialization Sequence

In a production environment, taking 5+ seconds to spin up an inference worker is unacceptable during auto-scaling events. LlamaForge is designed around strict latency thresholds during process initialization.

## 1. Fast Cold-Start Path

The cold-start path executes when a new pod/process launches with an empty host memory cache and no previous state.

### Execution Sequence:
1. **MMap Model Weights (Deferrable OS logic)**: 
   Use `mmap(MAP_SHARED)` for GGUF parsing. The OS block cache manages physical loading lazily on page faults. This blocks the main thread for <10ms to map headers, allowing the `ModelStore` to enter a ready state immediately.
2. **Context Preallocation (Latency-Critical, <20ms)**:
   Avoid dynamic tensor graph allocations across generations. Preallocate compute buffers in `InferenceContext` based on `max_batch_size`.
3. **Thread Pool Spin-Up (Latency-Critical, <50ms)**:
   Initialize `ggml_threadpool` and pin execution cores immediately upon process start. This ensures the scheduler avoids thread creations during the first request pipeline.
4. **Metadata Preparation (Latency-Critical, <5ms)**:
   Load necessary configs (EOS tokens, vocabulary).

### Summary of Latency
| Step | Criticality | Budget |
|---|---|---|
| Load Config / Vocabulary | High | < 5ms |
| Model Header MMap | High | < 10ms |
| Compute Scratch Prealloc (`InferenceContext`) | High | < 20ms |
| Thread Pool Initialization | High | < 15ms |
| Model Weight Page Faults | Deferrable | Async (OS driven) |
*Cold start Time to First Request Ready: ~50ms*

---

## 2. Warm-Start & Zero-Downtime Restarts

Warm-start involves launching an `InferenceContext` inside a pre-warmed OS (i.e. model data is already in page cache) or restoring user `SessionState` after a host daemon restart.

### Cached Runtime State
LlamaForge achieves sub-millisecond session resumption by decoupling user states from transient hardware context. 
Every `SessionState` implements a serialize/restore path to an external snapshot file or fast Redis store.

### Mechanism (`*.forge_snapshot`):
- **On graceful shutdown / Crash**: The running `SessionState` (Token history, exact current block ID, and current POS) is serialized to an mmap-backed `/dev/shm` IPC shard or a disk file (`.forge_snapshot`).
- **On restart**: 
  1. The new LlamaForge process spins up.
  2. Main process detects `snapshot_id=1234.forge_snapshot`.
  3. `SessionState` is reconstructed. Notice: the *physical KV Cache memory* must either be recomputed (Prefill replay) or if stored in shared memory (`/dev/shm`), re-attached instantly.
  4. The restored `SessionState` is attached to a free `InferenceContext` and generation continues seamlessly from the restored `current_pos_`.
