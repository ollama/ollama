// ============================================================================
// ggml_ext.h — C API Header for OLLaMA/llama.cpp Extension Integration
// Include this in ggml-cuda/hip.cu or ggml-vulkan.cpp to call extensions
// ============================================================================

#ifndef GGML_EXT_H
#define GGML_EXT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// HIP Extensions (libggml_hip_ext.so)
// ============================================================================

// --- Paged KV Cache ---
void* paged_kv_init(int num_layers, int num_heads, int head_dim, void* stream);
int paged_kv_alloc_sequence(void* mgr, int seq_id, int n_tokens, int n_layers);
int paged_kv_get_pages(void* mgr, int seq_id, int layer, int token_pos,
                       void** k_ptr, void** v_ptr, int* offset);
void paged_kv_free_sequence(void* mgr, int seq_id);
void paged_kv_get_stats(void* mgr, int* total, int* used, int* locked);

// --- RoPE Cache ---
void rope_cache_init(int head_dim, float theta, float scale, int use_yarn, void* stream);
void rope_apply_cached(void* q, void* k, int num_tokens, int num_heads, int head_dim,
                       int start_pos, int use_yarn, void* stream);

// --- MoE Routing ---
void moe_gate_topk(const float* gate_logits, float* expert_weights, int* expert_indices,
                   int num_tokens, int num_experts, void* stream);

// --- Async Upload ---
void async_upload_init(size_t max_size);
void* async_upload_begin();
void* async_upload_commit(size_t size);
void async_upload_sync();

// --- Quantized KV (Q8_0) ---
void* q8_0_kv_init(int num_heads, int head_dim, int max_tokens, void* stream);
void q8_0_kv_quantize(void* cache, const void* k_src, const void* v_src, int num_tokens, void* stream);
void q8_0_kv_attention(void* cache, const void* query, float* scores,
                      int num_tokens, int query_head, void* stream);
void q8_0_kv_free(void* cache);

// --- Split-K Matmul ---
void splitk_init(void* stream);
void splitk_gemm(void* A, void* B, void* C, int M, int N, int K,
                 int transA, int transB);

// --- K-Quant Dequant ---
void dequantize_q4_k(const void* blocks, void* output, int num_blocks, void* stream);

// --- Speculative Decoding ---
void speculative_init(int gram_size);
void speculative_learn(const int* tokens, int len);
int speculative_predict(const int* recent_tokens, int n);

// --- Persistent Batching ---
void persistent_batch_init(int max_tokens, void* stream);
void persistent_batch_reset();
void persistent_batch_add(int token, int pos, int seq_id, int want_logits);
void* persistent_batch_get_token_ptr();
void* persistent_batch_get_pos_ptr();
void* persistent_batch_get_logits_ptr();
int persistent_batch_size();

// ============================================================================
// Vulkan Extensions (libggml_vulkan_ext.so)
// ============================================================================

// Opaque Vulkan handles (use void* to avoid needing vulkan.h everywhere)
typedef void* VkDevice;
typedef void* VkPhysicalDevice;
typedef void* VkBuffer;
typedef void* VkCommandBuffer;
typedef void* VkDescriptorSet;
typedef void* VkDeviceMemory;
typedef uint32_t VkBool32;

// --- Async Upload ---
void vulkan_async_upload_init(VkDevice device, VkPhysicalDevice phys, uint64_t max_size);
void* vulkan_async_upload_begin();
void vulkan_async_upload_commit(VkBuffer dst, uint64_t size);
void vulkan_async_upload_sync();

// --- Subgroup Ops ---
void vulkan_subgroup_init(VkPhysicalDevice phys);
int vulkan_subgroup_get_size();

// --- Pipelined Scheduler ---
void vulkan_scheduler_init(VkDevice device, uint32_t queue_family);
VkCommandBuffer vulkan_scheduler_acquire();
void vulkan_scheduler_submit();
void vulkan_scheduler_drain();
int vulkan_scheduler_is_idle();

// --- Persistent Descriptors ---
void vulkan_descriptor_init(VkDevice device);
VkDescriptorSet vulkan_descriptor_acquire();
void vulkan_descriptor_update_buffer(VkDescriptorSet set, int binding, VkBuffer buffer, uint64_t size);

// --- Memory Pools ---
void vulkan_memory_pool_init(VkDevice device, VkPhysicalDevice phys,
                            uint64_t device_size, uint64_t staging_size);
uint64_t vulkan_memory_pool_alloc_device(uint64_t size, uint64_t alignment);
uint64_t vulkan_memory_pool_alloc_staging(uint64_t size, uint64_t alignment);
void vulkan_memory_pool_reset();
VkDeviceMemory vulkan_memory_pool_get_device_mem();
VkDeviceMemory vulkan_memory_pool_get_staging_mem();
void* vulkan_memory_pool_get_staging_ptr();

#ifdef __cplusplus
}
#endif

#endif // GGML_EXT_H
