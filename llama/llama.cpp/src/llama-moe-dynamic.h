#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// MoE Dynamic Loading API for Ollama
// Implements memory optimization techniques for Mixture-of-Experts models
// as described in GitHub issue #11005

struct llama_model;
struct llama_context;

// MoE expert loading parameters
struct llama_moe_hparams {
    int32_t n_experts;              // Total number of experts
    int32_t n_experts_used;         // Number of experts used per token
    size_t  vram_budget;            // Available VRAM for experts (bytes)
    size_t  cpu_budget;             // Available CPU memory for experts (bytes)
    int32_t cache_timeout_minutes;  // Expert cache timeout in minutes
    bool    enable_offloading;      // Enable expert offloading to CPU/disk
    bool    enable_sparse_compute;  // Enable sparse computation optimizations
    const char* eviction_policy;   // Eviction policy: "lru", "lfu", "random"
};

// MoE runtime statistics
struct llama_moe_stats {
    size_t  vram_usage;        // Current VRAM usage by experts
    size_t  cpu_usage;         // Current CPU usage by experts  
    size_t  disk_usage;        // Current disk usage by experts
    int32_t cache_hits;        // Number of cache hits
    int32_t cache_misses;      // Number of cache misses
    int32_t total_experts;     // Total number of experts
    int32_t loaded_experts;    // Currently loaded experts
    double  hit_rate;          // Cache hit rate (0.0 - 1.0)
};

// Expert loading callback - called when experts need to be loaded dynamically
// Returns true if experts were successfully made available
typedef bool (*llama_moe_expert_loader_t)(const int32_t* expert_ids, int32_t num_experts, int32_t priority, void* user_data);

// Initialize MoE dynamic loading system
// model: The llama model (can be NULL for standalone initialization)  
// hparams: MoE hyperparameters (can be NULL for defaults)
// Returns true if initialization was successful
bool llama_moe_dynamic_init_c(const struct llama_model* model, const struct llama_moe_hparams* hparams);

// Cleanup MoE dynamic loading system
void llama_moe_dynamic_cleanup_c(void);

// Enable/disable MoE dynamic loading
void llama_moe_dynamic_set_enabled_c(bool enabled);

// Check if MoE dynamic loading is enabled
bool llama_moe_dynamic_is_enabled_c(void);

// Set memory limits for expert management
void llama_moe_dynamic_set_memory_limits_c(size_t vram_limit, size_t cpu_limit);

// Get current MoE optimization statistics
struct llama_moe_stats llama_moe_dynamic_get_stats_c(void);

// Register expert loader callback (called from Go)
void llama_moe_dynamic_set_expert_loader_c(llama_moe_expert_loader_t loader, void* user_data);

// Request specific experts to be loaded (called from inference)
// expert_ids: Array of expert IDs to load
// num_experts: Number of experts in the array
// priority: Loading priority (higher = more urgent)
// Returns true if request was successful (experts available or loading)
bool llama_moe_dynamic_request_experts_c(const int32_t* expert_ids, int32_t num_experts, int32_t priority);

// Mark experts as actively used (for LRU tracking)
void llama_moe_dynamic_touch_experts_c(const int32_t* expert_ids, int32_t num_experts);

// Cleanup inactive experts (called periodically)
void llama_moe_dynamic_cleanup_inactive_c(void);

// Get expert location information
// expert_id: Expert ID to query
// Returns: 0=VRAM, 1=CPU, 2=Disk, -1=Unknown/Error
int32_t llama_moe_dynamic_get_expert_location_c(int32_t expert_id);

// Default MoE hyperparameters
struct llama_moe_hparams llama_moe_default_hparams(void);

#ifdef __cplusplus
}
#endif