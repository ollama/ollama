//go:build mpi

/**
 * llama.cpp - git 8183159cf3def112f6d1fe94815fce70e1bffa12
 *
 * MIT License
 *
 * Copyright (c) 2023 Georgi Gerganov
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_mpi_context;

void ggml_mpi_backend_init(void);
void ggml_mpi_backend_free(void);

struct ggml_mpi_context * ggml_mpi_init(void);
void ggml_mpi_free(struct ggml_mpi_context * ctx);

int ggml_mpi_rank(struct ggml_mpi_context * ctx);

void ggml_mpi_eval_init(
        struct ggml_mpi_context * ctx_mpi,
                            int * n_tokens,
                            int * n_past,
                            int * n_threads);

void ggml_mpi_graph_compute_pre(
        struct ggml_mpi_context * ctx_mpi,
             struct ggml_cgraph * gf,
                            int   n_layers);

void ggml_mpi_graph_compute_post(
        struct ggml_mpi_context * ctx_mpi,
             struct ggml_cgraph * gf,
                            int   n_layers);

#ifdef __cplusplus
}
#endif
