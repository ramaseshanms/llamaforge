# Deterministic Memory Management Strategy for llama.cpp Inference

In highly constrained environments (edge devices or multi-tenant hosting), relying on dynamic memory allocation during the critical path of an autoregressive decode loop introduces unacceptable jitter, fragmentation, and potential Out-Of-Memory (OOM) deaths.

This document details the deterministic, fixed-size arena allocation model designed for LlamaForge.

---

## 1. Allocation Model: Fixed-Size Arena

Traditional C++ `new`/`delete` and `malloc`/`free` are strictly forbidden once the server completes initialization. We replace them with a Preallocated Arena mapped over `ggml_backend`.

### Component Allocation Map
| Component | Phase | Allocation Strategy | Memory Ceiling |
|---|---|---|---|
| **Weights Tensors** (`llama_model`) | Init (Cold) | `mmap(MAP_SHARED)` backed by OS page cache. | Varies by model (e.g., 4GB for Llama3 8B Q4). Gracefully limited by system RAM. |
| **KV Cache** (`llama_kv_cache`) | Init (Warmup) | Fixed physical slab allocated statically on device/host. | Exact size computed via `max_context * batch_size * layer_bytes`. Hard limit. |
| **Compute Scratch Buffer** (`ggml_context`) | Init (Warmup) | Single Fixed-size Arena buffer. Reused for all forward passes. | Pre-calculated peak graph size per exact configured batch config. (e.g., 256MB). |
| **Token History & User State** | Inference | Pre-reserved `std::vector` (capacity = `max_context`). | Lightweight (~16KB per session). |

### Hard Ceilings and Behavior
If an incoming request exceeds the configured `max_context` or batch limits:
- **No dynamic growth occurs.** The `std::bad_alloc` vector is completely removed.
- **Behavior**: The `InferenceContext` throws a strictly typed error `LlamaResourceExhaustedException`. The `SessionState` immediately transitions into a `PAUSED` backoff state to try again when another slot clears.

---

## 2. ggml Tensor Lifetimes: Autoregressive Decode Step

An autoregressive decode step evaluates exactly one token. It requires allocating a computational graph that traverses attention and MLP layers.

### Tensor Allocation Trace
During a step (`InferenceContext::EvaluateNextToken`), `ggml_cgraph` is built.
1. **Model Weights**: Read-only static bounds (already `mmap`-ed).
2. **Intermediate Activations (Q, K, V, RoPE)**: 
   - These are dynamically materializing nodes in the graph.
   - However, in our system, they are requested via the **Arena Allocator**. We bump a single pointer `scratch_buf->data += size`.
3. **KV Cache Storage**:
   - The computed `K` and `V` nodes are directly written `memcpy` into the persistent KV Cache slab at the offset determined by `current_pos_`.

### Stack Allocation & Lazy Materialization
- We apply `ggml_set_scratch()` to map transient operation outputs (like SiLU activations) strictly onto our Arena. 
- *Lazy Materialization*: Nodes such as `RoPE` frequencies are computed on-the-fly inside kernel registers rather than physically allocating a tensor holding them globally.

### Tensor Reuse and Aliasing
Because intermediate activation tensors (e.g. output from Layer N-1 to Layer N) do not logically exist simultaneously across the entire graph, they occupy overlapping offsets inside the scratch Arena. 
- **Risk**: Aliasing bugs map two logically disparate nodes to the same RAM address prematurely.
- **Mitigation**: Using `ggml_gallocr` (graph allocator) to pre-calculate the precise lifespan of intermediate nodes mathematically before the forward pass runs. The graph allocator resolves exact byte-offsets guaranteeing disjoint memory only during active execution windows.

---

## 3. Long-Session Stability & Fragmentation Risk

Processing a continuous stream of tokens for millions of steps (e.g., massive context multi-turn RP or Agentic loops) risks heap degradation.

### Fragmentation Analysis
If we allowed standard `std::string` or `std::vector` to hold conversation history dynamically, repeated appends up to 128K context window would trigger continuous `realloc` operations, fragmenting the heap and blowing up `RSS` arbitrarily.
- **LlamaForge's Fix**: By allocating the precise `SessionState` objects at max bounds up front and relying *exclusively* on the fixed Arena, 1 million decode steps have exactly 0 heap allocations. **Fragmentation risk is mathematically 0%**.

### KV Cache Compaction (Slab Strategy)
When managing multiple concurrent sessions, users disconnecting mid-stream leaves "holes" in the KV Cache.
Rather than moving massive blocks of memory (expensive), we use a **Paged KV Slab** (similar to vLLM logic):
1. **Logical vs Physical**: The `llama_kv_cache` is subdivided into generic 16-token memory blocks (Slabs).
2. **Scatter-Gather**: The attention kernels access K and V by routing through a page table.
3. This is fully compatible with `ggml_backend` custom operations by writing a custom scatter-gather kernel mapping physical chunks over the logical tensor sequence.

### Metrics to Detect Pressure
1. **Arena High-Water Mark (`arena_watermark_bytes`)**:
   Tracks the maximum graph size required. If a query hits 95% of the pre-allocated arena size, a metric triggers a warning before any memory faults occur.
2. **Slab Occupancy Ratio `%`**: 
   (Active KV Pages / Total KV Pages). Alerts upstream routing to stop sending requests as we approach eviction pressure.
3. **Page Fault Miss Rate `pf_rate_sec`**:
   Ensures the OS isn't silently paging out `mmap` blocks, ensuring 100% deterministic latency.
