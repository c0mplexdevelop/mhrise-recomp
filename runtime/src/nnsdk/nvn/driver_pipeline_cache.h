#pragma once

#include "nnsdk/nvn/vk.h"
#include <atomic>
#include <chrono>
#include <shared_mutex>
#include <utility>
#include <vector>

namespace nvn {

class DriverPipelineCache {
public:
    void load();
    void save(uint64_t frame);
    void merge(VkPipelineCache source, uint64_t created);
    std::vector<uint8_t> take_warm_data() { return std::move(warm_cache_data_); }
    void note_prefetch() { ++warm_new_; }
    void note_created(uint64_t frame) {
        ++pipelines_created_;
        last_pipeline_frame_ = frame;
    }
    uint64_t created() const { return pipelines_created_; }
    bool reading() const { return cache_reading_.load(std::memory_order_relaxed); }

    template<class F> VkResult compile(F&& build) {
        std::shared_lock lock(pipeline_cache_lock_);
        return build(pipeline_cache_);
    }

private:
    VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
    std::shared_mutex pipeline_cache_lock_;
    std::vector<uint8_t> warm_cache_data_;
    std::atomic<uint64_t> warm_new_{0}, pipelines_saved_{0};
    std::atomic<bool> cache_saving_{false}, cache_reading_{false};
    uint64_t pipelines_created_ = 0, last_pipeline_frame_ = 0;
    std::chrono::steady_clock::time_point pipelines_saved_at_ = std::chrono::steady_clock::now();
};

}  // namespace nvn
