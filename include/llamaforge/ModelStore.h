#pragma once

#include <string>
#include <memory>
#include <vector>

// Forward declarations
struct llama_model;

namespace llamaforge {

struct ModelFormat {
    bool is_gguf;
    size_t param_count;
    size_t embed_dim;
    size_t vocab_size;
    bool has_lora_adapters;
};

/**
 * @brief Represents the lifetime and metadata of a loaded LLM.
 *
 * ModelStore is responsible for loading the model weights (e.g. from a GGUF file),
 * mmap-ing the heavy tensors into RAM, and managing the memory allocated for
 * the static computational graph structures (llama_model).
 *
 * Thread-safety: ModelStore is read-only after initialization and can be safely 
 * shared across multiple worker threads and concurrent sessions.
 */
class ModelStore {
public:
    explicit ModelStore(const std::string& model_path);
    ~ModelStore();

    // Delete copy/move to enforce strict singleton-like lifetime per model file
    ModelStore(const ModelStore&) = delete;
    ModelStore& operator=(const ModelStore&) = delete;
    ModelStore(ModelStore&&) = delete;
    ModelStore& operator=(ModelStore&&) = delete;

    /// Loads the model synchronously (cold start mapping)
    bool Load();
    
    /// Evicts the model from memory without destroying the object 
    void Evict();

    /// Returns the raw llama.cpp model pointer
    llama_model* GetRawModel() const { return model_; }

    /// Returns basic metadata about the loaded model
    const ModelFormat& GetMetadata() const { return metadata_; }

    const std::string& GetPath() const { return path_; }

private:
    std::string path_;
    llama_model* model_ = nullptr;
    ModelFormat metadata_{};
    bool is_loaded_ = false;
};

} // namespace llamaforge
