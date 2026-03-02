# KV-Cache Design & Paging Strategy

In streaming inference for long-context language models, the Key-Value (KV) cache becomes the primary memory bottleneck, scaling linearly with batch size and context length. This document details the LlamaForge KV-cache architecture designed for multi-session edge deployment under fixed-memory budgets.

---

## 1. KV-Cache Memory Layout & Scaling

LlamaForge avoids contiguous monolithic buffers for the KV-cache, which suffer from external fragmentation when user sessions disconnect or abort. Instead, we use a **Paged KV Layout** (inspired by vLLM) adapted for `ggml`.

### Exact Memory Layout (Per Token)
The memory required to cache a single token's KV states depends on the model's architecture. For a standard LLaMA-3 (Grouped-Query Attention with 8 KV heads, 128 head-dim, 32 layers, FP16 precision):

- **Key Size per token**: $8 \text{ heads} \times 128 \text{ dim} \times 32 \text{ layers} \times 2 \text{ bytes (FP16)} = 65,536 \text{ bytes (64 KB)}$
- **Value Size per token**: $65,536 \text{ bytes (64 KB)}$
- **Total per token**: **128 KB**

### Memory Growth & Alignment
- **Growth Function**: $M(c) = 128\text{KB} \times c$ where $c$ is the context length. 
  *A single 8,192 token session requires ~1 GB of KV-cache.*
- **Block Organization**: Cache is divided into fixed-size **Blocks** (e.g., 16 tokens).
  *Block Size*: $16 \times 128\text{KB} = 2\text{MB}$.
- **Alignment & Cache Lines**: The 2MB blocks are aligned to `256-byte` boundaries optimized for AVX-512 / ARM NEON vectorization in `ggml` custom kernels. Keys and Values are kept in separate contiguous slab arrays to maximize CPU cache-line usage during the attention dot-product `(Q @ K.T)`.

---

## 2. Paging and Eviction Policies

When the fixed KV pool nears 100% capacity, LlamaForge dynamically swaps or drops blocks rather than crashing.

### The Paging Mechanism
The `SessionState` no longer owns physical memory. It holds a **Virtual Block Table** mapping logical token indices `[0..15]` to physical `Block_ID = 42`.
When memory pressure hits, physical blocks are moved back to host RAM (if offloading from GPU/NPU) or evicted entirely.

### Eviction Policies
1. **LRU (Idle Session Swapping)**:
   - If a user session has not requested a token in > 30 seconds (`status == PAUSED`), its physical KV blocks are evicted to disk or deleted.
2. **Token-Window Eviction (Streaming Context)**:
   - For long-running agent loops (e.g., millions of tokens), LlamaForge supports *Local Window Attention*. We evict the oldest KV blocks (e.g., tokens `[0..2048]`), keeping only the most recent tokens in physical mapping, allowing infinite generation.
3. **Session-Priority Dropping**:
   - If strict OOM is imminent, low-priority sessions (labeled via API) are unmapped entirely. Their generation fails gracefully with HTTP 429 backoff.

### Correctness Guarantees (Suffix Replay)
If a PAUSED session's KV-cache is evicted, the user's `SessionState` survives because it retains the original text/token IDs. 
When the user resumes, LlamaForge allocates fresh Physical Blocks and triggers a hidden, high-speed **Prefill Replay** over the token history to silently rebuild the KV states before decoding the next token. This ensures absolute algebraic correctness.

---

## 3. Multi-Session Isolation & Quotas

A rogue session sending a 128K context prompt must not crash or starve concurrent 2K context sessions.

### Quota Enforcement
At session instantiation, the `InferenceContext` scheduler assigns a **Maximum Block Quota** based on the user's tier.
- A single session cannot allocate Physical Blocks beyond its quota limit.
- If $Session_A$ requests block 101 when its quota is 100, the `InferenceContext` intercepts the allocation and triggers the *Token-Window Eviction* (dropping its own oldest block to make room for block 101).
- **Isolation**: $Session_A$'s growth is strictly clamped. It will never steal blocks from $Session_B$'s pre-allocated mapping limits.

### Safe Teardown
When a user disconnects:
1. `SessionState(ID=X)` calls `ReleaseContext()`.
2. The Virtual Block Table iterates over its mapped Physical Blocks.
3. The global `KVArenaManager` marks those Block IDs as `FREE_CLEAN` atomically.
4. Memory is zeroed lazily on the next allocation by a new session, eliminating tear-down latency bottlenecks.
