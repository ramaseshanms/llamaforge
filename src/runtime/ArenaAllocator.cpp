#include "llamaforge/ArenaAllocator.h"
#include <sstream>

namespace llamaforge {

ArenaAllocator::ArenaAllocator(size_t total_bytes) 
    : buffer_(total_bytes, 0), capacity_(total_bytes), used_(0) {
}

void* ArenaAllocator::Allocate(size_t bytes, size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("Arena alignment must be a power of two");
    }

    // Calculate aligned offset
    size_t current_ptr = reinterpret_cast<size_t>(buffer_.data() + used_);
    size_t offset = (alignment - (current_ptr % alignment)) % alignment;
    size_t total_required = bytes + offset;

    if (used_ + total_required > capacity_) {
        std::ostringstream oss;
        oss << "Arena Capacity Exceeded: Need " << total_required 
            << " bytes but only " << (capacity_ - used_) << " available.";
        throw LlamaResourceExhaustedException(oss.str());
    }

    void* aligned_ptr = buffer_.data() + used_ + offset;
    used_ += total_required;
    
    return aligned_ptr;
}

void ArenaAllocator::Reset() {
    used_ = 0;
}

} // namespace llamaforge
