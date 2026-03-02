#include "llamaforge/ModelStore.h"

namespace llamaforge {

ModelStore::ModelStore(const std::string& model_path) : path_(model_path) {}
ModelStore::~ModelStore() { Evict(); }

bool ModelStore::Load() {
    is_loaded_ = true;
    return true;
}

void ModelStore::Evict() {
    is_loaded_ = false;
}

} // namespace llamaforge
