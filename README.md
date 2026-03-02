# LlamaForge

**LlamaForge** is a production-grade C++ inference runtime built around `llama.cpp` and `ggml`. 
It is designed by and for Memory Systems and AI Inference Engineers to deploy LLMs in memory-constrained, low-latency edge environments with 100% deterministic resource control.

## Motivation
Standard inference runtimes often rely on OS-level page eviction, dynamic allocations during decoding, or non-deterministic garbage collection. LlamaForge strips this back:
- **Zero dynamic allocations** during the autoregressive hot loop.
- **Fail-safe recovery**: hardware execution contexts are strictly isolated from user conversation states (`SessionState`).
- **Sub-50ms cold starts**: through memory-mapped weight pages and pre-pinned thread pools.

## Architecture

LlamaForge separates concerns into three distinct C++ layers:
1. `ModelStore`: Handles the lifetime of `mmap`-ed weights and static `ggml` graphs.
2. `SessionState`: Logic-only layer maintaining KV cache sequence slots and conversation token history.
3. `InferenceContext`: An ephemeral, recyclable hardware execution block containing scratch buffers and thread bindings.

For detailed architecture diagrams and memory ownership rules, see the [Architecture Docs](docs/architecture.md).

## Getting Started

### Prerequisites
- Linux/POSIX environment
- CMake 3.10+
- GCC/Clang with C++17 support

### Building
LlamaForge is built via standard CMake:

```bash
mkdir build
cd build
cmake ..
make -j4
```
