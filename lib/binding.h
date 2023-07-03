#ifdef __cplusplus
#include <vector>
#include <string>
extern "C" {
#endif

#include <stdbool.h>

extern unsigned char tokenCallback(void *, char *);

int load_state(void *ctx, char *statefile, char*modes);

int eval(void* params_ptr, void *ctx, char*text);

void save_state(void *ctx, char *dst, char*modes);

void* load_model(const char *fname, int n_ctx, int n_seed, bool memory_f16, bool mlock, bool embeddings, bool mmap, bool low_vram, bool vocab_only, int n_gpu, int n_batch, const char *maingpu, const char *tensorsplit, bool numa);

int get_embeddings(void* params_ptr, void* state_pr, float * res_embeddings);

int get_token_embeddings(void* params_ptr, void* state_pr,  int *tokens, int tokenSize, float * res_embeddings);

void* llama_allocate_params(const char *prompt, int seed, int threads, int tokens,
                            int top_k, float top_p, float temp, float repeat_penalty, 
                            int repeat_last_n, bool ignore_eos, bool memory_f16, 
                            int n_batch, int n_keep, const char** antiprompt, int antiprompt_count,
                            float tfs_z, float typical_p, float frequency_penalty, float presence_penalty, int mirostat, float mirostat_eta, float mirostat_tau, bool penalize_nl, const char *logit_bias, const char *session_file, bool prompt_cache_all, bool mlock, bool mmap, const char *maingpu, const char *tensorsplit , bool prompt_cache_ro);

void llama_free_params(void* params_ptr);

void llama_binding_free_model(void* state);

int llama_predict(void* params_ptr, void* state_pr, char* result, bool debug);

#ifdef __cplusplus
}


std::vector<std::string> create_vector(const char** strings, int count);
void delete_vector(std::vector<std::string>* vec);
#endif
