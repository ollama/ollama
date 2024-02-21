#if defined(LLAMA_SERVER_LIBRARY)
#ifndef LLAMA_SERVER_H
#define LLAMA_SERVER_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

int __main(int argc, char **argv);

// This exposes extern C entrypoints into the llama_server
// To enable the server compile with LLAMA_SERVER_LIBRARY

#ifdef __cplusplus
extern "C" {
#endif
typedef struct ext_server_resp {
  int id;          // < 0 on error
  size_t msg_len;  // caller must allocate msg and set msg_len
  char *msg;
} ext_server_resp_t;

// Allocated and freed by caller
typedef struct ext_server_lora_adapter {
  char *adapter;
  float scale;
  struct ext_server_lora_adapter *next;
} ext_server_lora_adapter_t;

// Allocated and freed by caller
typedef struct ext_server_params {
  char *model;
  uint32_t n_ctx;         // token context window, 0 = from model
  uint32_t n_batch;       // prompt processing maximum batch size
  uint32_t n_threads;     // number of threads to use for generation
  int32_t n_parallel;     // number of parallel sequences to decodewra
  float rope_freq_base;   // RoPE base frequency, 0 = from model
  float rope_freq_scale;  // RoPE frequency scaling factor, 0 = from model
  bool memory_f16;        // use f16 instead of f32 for memory kv
  int32_t n_gpu_layers;  // number of layers to store in VRAM (-1 - use default)
  int32_t main_gpu;      // the GPU that is used for scratch and small tensors
  bool use_mlock;        // force system to keep model in RAM
  bool use_mmap;         // use mmap if possible
  int numa;              // attempt optimizations that help on some NUMA systems
  bool embedding;        // get only sentence embedding
  ext_server_lora_adapter_t *lora_adapters;
  char *mmproj;
  bool verbose_logging;  // Enable verbose logging of the server
} ext_server_params_t;

typedef struct ext_server_task_result {
  int id;
  bool stop;
  bool error;
  char *json_resp;  // null terminated, memory managed by ext_server
} ext_server_task_result_t;

// Initialize the server once per process
// err->id = 0 for success and err->msg[0] = NULL
// err->id != 0 for failure, and err->msg contains error message
void llama_server_init(ext_server_params_t *sparams, ext_server_resp_t *err);

// Run the main loop, called once per init
void llama_server_start();
// Stop the main loop and free up resources allocated in init and start.  Init
// must be called again to reuse
void llama_server_stop();

// json_req null terminated string, memory managed by caller
// resp->id >= 0 on success (task ID)
// resp->id < 0 on error, and resp->msg contains error message
void llama_server_completion(const char *json_req, ext_server_resp_t *resp);

// Caller must call llama_server_release_task_result to free resp->json_resp
void llama_server_completion_next_result(const int task_id,
                                         ext_server_task_result_t *result);
void llama_server_completion_cancel(const int task_id, ext_server_resp_t *err);
void llama_server_release_task_result(ext_server_task_result_t *result);

// Caller must call llama_server_releaes_json_resp to free json_resp if err.id <
// 0
void llama_server_tokenize(const char *json_req, char **json_resp,
                           ext_server_resp_t *err);
void llama_server_detokenize(const char *json_req, char **json_resp,
                             ext_server_resp_t *err);
void llama_server_embedding(const char *json_req, char **json_resp,
                            ext_server_resp_t *err);
void llama_server_release_json_resp(char **json_resp);

#ifdef __cplusplus
}
#endif

#endif
#endif  // LLAMA_SERVER_LIBRARY