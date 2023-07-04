// MIT License

// Copyright (c) 2023 go-skynet authors

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifdef __cplusplus
#include <string>
#include <vector>
extern "C" {
#endif

#include <stdbool.h>

extern unsigned char tokenCallback(void *, char *);

int eval(void *p, void *c, char *text);

void *load_model(const char *fname, int n_ctx, int n_seed, bool memory_f16,
                 bool mlock, bool embeddings, bool mmap, bool low_vram,
                 bool vocab_only, int n_gpu, int n_batch, const char *maingpu,
                 const char *tensorsplit, bool numa);

void *llama_allocate_params(
    const char *prompt, int seed, int threads, int tokens, int top_k,
    float top_p, float temp, float repeat_penalty, int repeat_last_n,
    bool ignore_eos, bool memory_f16, int n_batch, int n_keep,
    const char **antiprompt, int antiprompt_count, float tfs_z, float typical_p,
    float frequency_penalty, float presence_penalty, int mirostat,
    float mirostat_eta, float mirostat_tau, bool penalize_nl,
    const char *logit_bias, const char *session_file, bool prompt_cache_all,
    bool mlock, bool mmap, const char *maingpu, const char *tensorsplit,
    bool prompt_cache_ro);

void llama_free_params(void *params_ptr);

void llama_binding_free_model(void *ctx);

int llama_predict(void *params_ptr, void *state_pr, char *result, bool debug);

#ifdef __cplusplus
}

#endif
