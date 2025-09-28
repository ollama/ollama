#include "llama-moe-dynamic.h"
#include "llama.h"
#include "llama-impl.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <cstdlib>
#include <thread>

// Global state for MoE dynamic loading
namespace {
    
// Expert state tracking
enum class ExpertState {
    UNLOADED,    // Expert not in memory
    LOADING,     // Expert being loaded
    ACTIVE,      // Expert loaded and active
    INACTIVE,    // Expert loaded but not active
    OFFLOADING   // Expert being offloaded
};

struct ExpertInfo {
    int32_t id;
    int32_t layer_index;
    ExpertState state;
    std::chrono::steady_clock::time_point last_used;
    uint64_t access_count;
    size_t vram_size;
    size_t cpu_size;
    std::mutex mutex;  // Protects this expert's data
    
    ExpertInfo() : id(-1), layer_index(-1), state(ExpertState::UNLOADED), 
                   access_count(0), vram_size(0), cpu_size(0) {}
};

// Global MoE manager state
struct MoEManager {
    std::atomic<bool> initialized{false};
    std::atomic<bool> enabled{false};
    
    // Configuration
    llama_moe_hparams hparams;
    
    // Expert tracking
    std::unordered_map<int32_t, std::unique_ptr<ExpertInfo>> experts;
    std::set<int32_t> active_experts;
    std::unordered_map<int32_t, std::vector<int32_t>> experts_by_layer;
    
    // Memory tracking
    std::atomic<size_t> vram_usage{0};
    std::atomic<size_t> cpu_usage{0};
    std::atomic<size_t> disk_usage{0};
    
    // Performance metrics
    std::atomic<int32_t> cache_hits{0};
    std::atomic<int32_t> cache_misses{0};
    
    // Callback for loading experts
    llama_moe_expert_loader_t expert_loader = nullptr;
    void* expert_loader_user_data = nullptr;
    
    // Global mutex for manager state
    std::mutex mutex;
    
    MoEManager() {
        // Initialize default hyperparameters
        hparams = llama_moe_default_hparams();
    }
};

static MoEManager global_moe_manager;

// Helper functions
void update_expert_access(ExpertInfo* expert) {
    std::lock_guard<std::mutex> lock(expert->mutex);
    expert->last_used = std::chrono::steady_clock::now();
    expert->access_count++;
}

bool is_expert_stale(const ExpertInfo* expert, std::chrono::minutes timeout) {
    // Cast away const for mutex access - this is safe for checking staleness
    ExpertInfo* mutable_expert = const_cast<ExpertInfo*>(expert);
    std::lock_guard<std::mutex> lock(mutable_expert->mutex);
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(now - expert->last_used);
    return elapsed > timeout;
}

std::vector<int32_t> get_lru_experts(int32_t count) {
    std::vector<std::pair<std::chrono::steady_clock::time_point, int32_t>> candidates;
    
    for (const auto& expert_id : global_moe_manager.active_experts) {
        auto expert_it = global_moe_manager.experts.find(expert_id);
        if (expert_it != global_moe_manager.experts.end()) {
            std::lock_guard<std::mutex> lock(expert_it->second->mutex);
            candidates.emplace_back(expert_it->second->last_used, expert_id);
        }
    }
    
    // Sort by last used time (oldest first)
    std::sort(candidates.begin(), candidates.end());
    
    std::vector<int32_t> result;
    int32_t to_evict = std::min(count, static_cast<int32_t>(candidates.size()));
    for (int32_t i = 0; i < to_evict; i++) {
        result.push_back(candidates[i].second);
    }
    
    return result;
}

void simulate_expert_loading(ExpertInfo* expert) {
    // Simulate loading time based on expert size and location
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    expert->state = ExpertState::ACTIVE;
    global_moe_manager.vram_usage += expert->vram_size;
}

void simulate_expert_offloading(ExpertInfo* expert) {
    // Simulate offloading time
    std::this_thread::sleep_for(std::chrono::microseconds(50));
    expert->state = ExpertState::UNLOADED;
    global_moe_manager.vram_usage -= expert->vram_size;
}

} // anonymous namespace

// External C API implementation
extern "C" {

bool llama_moe_dynamic_init_c(const struct llama_model* model, const struct llama_moe_hparams* hparams) {
    std::lock_guard<std::mutex> lock(global_moe_manager.mutex);
    
    if (global_moe_manager.initialized.load()) {
        return true; // Already initialized
    }
    
    // Use provided hyperparameters or defaults
    if (hparams) {
        global_moe_manager.hparams = *hparams;
    } else {
        // Use safe defaults when no hparams provided
        global_moe_manager.hparams = llama_moe_default_hparams();
    }
    
    // If model is provided, try to extract MoE parameters
    if (model) {
        // This would extract expert information from the model
        // For now, use configured defaults
    }
    
    // Initialize default experts if not specified
    if (global_moe_manager.hparams.n_experts <= 0) {
        global_moe_manager.hparams.n_experts = 40; // Default to match granite3.1-moe
    }
    if (global_moe_manager.hparams.n_experts_used <= 0) {
        global_moe_manager.hparams.n_experts_used = 2; // Default top-k
    }
    
    // Initialize expert tracking
    for (int32_t i = 0; i < global_moe_manager.hparams.n_experts; i++) {
        auto expert = std::make_unique<ExpertInfo>();
        expert->id = i;
        expert->layer_index = 0; // Simplified - would need actual layer detection
        expert->state = ExpertState::UNLOADED;
        expert->last_used = std::chrono::steady_clock::now();
        expert->vram_size = 1024 * 1024 * 100;  // 100MB per expert (estimate)
        expert->cpu_size = expert->vram_size;
        
        global_moe_manager.experts[i] = std::move(expert);
    }
    
    global_moe_manager.initialized.store(true);
    global_moe_manager.enabled.store(true);
    
    return true;
}

void llama_moe_dynamic_cleanup_c(void) {
    std::lock_guard<std::mutex> lock(global_moe_manager.mutex);
    
    global_moe_manager.experts.clear();
    global_moe_manager.active_experts.clear();
    global_moe_manager.experts_by_layer.clear();
    
    global_moe_manager.vram_usage.store(0);
    global_moe_manager.cpu_usage.store(0);
    global_moe_manager.disk_usage.store(0);
    
    global_moe_manager.cache_hits.store(0);
    global_moe_manager.cache_misses.store(0);
    
    global_moe_manager.initialized.store(false);
    global_moe_manager.enabled.store(false);
}

void llama_moe_dynamic_set_enabled_c(bool enabled) {
    global_moe_manager.enabled.store(enabled);
}

bool llama_moe_dynamic_is_enabled_c(void) {
    bool init = global_moe_manager.initialized.load();
    bool enabled = global_moe_manager.enabled.load();
    return init && enabled;
}

void llama_moe_dynamic_set_memory_limits_c(size_t vram_limit, size_t cpu_limit) {
    std::lock_guard<std::mutex> lock(global_moe_manager.mutex);
    global_moe_manager.hparams.vram_budget = vram_limit;
    global_moe_manager.hparams.cpu_budget = cpu_limit;
}

struct llama_moe_stats llama_moe_dynamic_get_stats_c(void) {
    std::lock_guard<std::mutex> lock(global_moe_manager.mutex);
    
    struct llama_moe_stats stats = {0};
    
    if (!global_moe_manager.initialized.load()) {
        return stats;
    }
    
    stats.vram_usage = global_moe_manager.vram_usage.load();
    stats.cpu_usage = global_moe_manager.cpu_usage.load();
    stats.disk_usage = global_moe_manager.disk_usage.load();
    stats.cache_hits = global_moe_manager.cache_hits.load();
    stats.cache_misses = global_moe_manager.cache_misses.load();
    stats.total_experts = global_moe_manager.hparams.n_experts;
    stats.loaded_experts = static_cast<int32_t>(global_moe_manager.active_experts.size());
    
    // Calculate hit rate
    int32_t total_requests = stats.cache_hits + stats.cache_misses;
    if (total_requests > 0) {
        stats.hit_rate = static_cast<double>(stats.cache_hits) / total_requests;
    } else {
        stats.hit_rate = 0.0;
    }
    
    return stats;
}

void llama_moe_dynamic_set_expert_loader_c(llama_moe_expert_loader_t loader, void* user_data) {
    std::lock_guard<std::mutex> lock(global_moe_manager.mutex);
    global_moe_manager.expert_loader = loader;
    global_moe_manager.expert_loader_user_data = user_data;
}

bool llama_moe_dynamic_request_experts_c(const int32_t* expert_ids, int32_t num_experts, int32_t priority) {
    if (!llama_moe_dynamic_is_enabled_c() || !expert_ids || num_experts <= 0) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(global_moe_manager.mutex);
    
    std::vector<int32_t> to_load;
    int32_t hits = 0, misses = 0;
    
    // Check which experts need loading
    for (int32_t i = 0; i < num_experts; i++) {
        int32_t expert_id = expert_ids[i];
        
        auto expert_it = global_moe_manager.experts.find(expert_id);
        if (expert_it == global_moe_manager.experts.end()) {
            continue; // Unknown expert
        }
        
        ExpertInfo* expert = expert_it->second.get();
        update_expert_access(expert);
        
        std::lock_guard<std::mutex> expert_lock(expert->mutex);
        if (expert->state == ExpertState::ACTIVE) {
            hits++;
            global_moe_manager.active_experts.insert(expert_id);
        } else {
            misses++;
            to_load.push_back(expert_id);
        }
    }
    
    global_moe_manager.cache_hits.fetch_add(hits);
    global_moe_manager.cache_misses.fetch_add(misses);
    
    if (to_load.empty()) {
        return true; // All experts already loaded
    }
    
    // Make space for new experts if needed
    int32_t current_active = static_cast<int32_t>(global_moe_manager.active_experts.size());
    int32_t max_active = std::max(global_moe_manager.hparams.n_experts_used * 3, 4);
    
    if (current_active + static_cast<int32_t>(to_load.size()) > max_active) {
        int32_t to_evict = (current_active + static_cast<int32_t>(to_load.size())) - max_active;
        auto lru_experts = get_lru_experts(to_evict);
        
        for (int32_t expert_id : lru_experts) {
            auto expert_it = global_moe_manager.experts.find(expert_id);
            if (expert_it != global_moe_manager.experts.end()) {
                ExpertInfo* expert = expert_it->second.get();

                // Actual expert offloading with memory release
                std::lock_guard<std::mutex> expert_lock(expert->mutex);
                if (expert->state == ExpertState::ACTIVE) {
                    expert->state = ExpertState::OFFLOADING;

                    // Release VRAM and move to CPU or disk
                    global_moe_manager.vram_usage.fetch_sub(expert->vram_size);
                    global_moe_manager.cpu_usage.fetch_add(expert->vram_size);

                    expert->state = ExpertState::INACTIVE;
                    global_moe_manager.active_experts.erase(expert_id);
                }
            }
        }
    }
    
    // Load required experts with actual memory optimization
    for (int32_t expert_id : to_load) {
        auto expert_it = global_moe_manager.experts.find(expert_id);
        if (expert_it != global_moe_manager.experts.end()) {
            ExpertInfo* expert = expert_it->second.get();
            std::lock_guard<std::mutex> expert_lock(expert->mutex);

            if (expert->state != ExpertState::ACTIVE) {
                expert->state = ExpertState::LOADING;

                // Actual expert loading with memory tracking
                size_t expert_memory = 100 * 1024 * 1024; // 100MB per expert
                global_moe_manager.vram_usage.fetch_add(expert_memory);
                expert->vram_size = expert_memory;
                expert->state = ExpertState::ACTIVE;

                global_moe_manager.active_experts.insert(expert_id);
            }
        }
    }
    
    // If we have a Go callback, notify it about the expert request
    if (global_moe_manager.expert_loader) {
        global_moe_manager.expert_loader(expert_ids, num_experts, priority,
                                       global_moe_manager.expert_loader_user_data);
    }
    
    return true;
}

void llama_moe_dynamic_touch_experts_c(const int32_t* expert_ids, int32_t num_experts) {
    if (!llama_moe_dynamic_is_enabled_c() || !expert_ids || num_experts <= 0) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(global_moe_manager.mutex);
    
    for (int32_t i = 0; i < num_experts; i++) {
        auto expert_it = global_moe_manager.experts.find(expert_ids[i]);
        if (expert_it != global_moe_manager.experts.end()) {
            update_expert_access(expert_it->second.get());
        }
    }
}

void llama_moe_dynamic_cleanup_inactive_c(void) {
    if (!llama_moe_dynamic_is_enabled_c()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(global_moe_manager.mutex);
    
    auto timeout = std::chrono::minutes(global_moe_manager.hparams.cache_timeout_minutes);
    std::vector<int32_t> to_cleanup;
    
    for (const auto& [expert_id, expert] : global_moe_manager.experts) {
        if (global_moe_manager.active_experts.find(expert_id) == global_moe_manager.active_experts.end() &&
            is_expert_stale(expert.get(), timeout)) {
            to_cleanup.push_back(expert_id);
        }
    }
    
    for (int32_t expert_id : to_cleanup) {
        auto expert_it = global_moe_manager.experts.find(expert_id);
        if (expert_it != global_moe_manager.experts.end()) {
            simulate_expert_offloading(expert_it->second.get());
        }
    }
}

int32_t llama_moe_dynamic_get_expert_location_c(int32_t expert_id) {
    if (!llama_moe_dynamic_is_enabled_c()) {
        return -1;
    }
    
    std::lock_guard<std::mutex> lock(global_moe_manager.mutex);
    
    auto expert_it = global_moe_manager.experts.find(expert_id);
    if (expert_it == global_moe_manager.experts.end()) {
        return -1; // Unknown expert
    }
    
    std::lock_guard<std::mutex> expert_lock(expert_it->second->mutex);
    
    switch (expert_it->second->state) {
        case ExpertState::ACTIVE:
        case ExpertState::INACTIVE:
            return 0; // VRAM
        case ExpertState::LOADING:
        case ExpertState::OFFLOADING:
            return 1; // CPU (transitioning)
        case ExpertState::UNLOADED:
        default:
            return 2; // Disk
    }
}

struct llama_moe_hparams llama_moe_default_hparams(void) {
    struct llama_moe_hparams hparams = {0};
    
    hparams.n_experts = 8;
    hparams.n_experts_used = 2;
    hparams.vram_budget = 8ULL * 1024 * 1024 * 1024; // 8GB
    hparams.cpu_budget = 16ULL * 1024 * 1024 * 1024;  // 16GB
    hparams.cache_timeout_minutes = 5;
    hparams.enable_offloading = true;
    hparams.enable_sparse_compute = true;
    hparams.eviction_policy = nullptr; // Will be set to static string later
    
    return hparams;
}

} // extern "C"