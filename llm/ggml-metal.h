//go:build darwin

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

// An interface allowing to compute ggml_cgraph with Metal
//
// This is a fully functional interface that extends ggml with GPU support for Apple devices.
// A similar interface can be created for other GPU backends (e.g. Vulkan, CUDA, OpenCL, etc.)
//
// How it works?
//
// As long as your program can create and evaluate a ggml_cgraph on the CPU, you can use this
// interface to evaluate the same graph on the GPU. Instead of using ggml_graph_compute(), you
// use ggml_metal_graph_compute() (or ggml_vulkan_graph_compute(), etc.)
//
// You only need to make sure that all memory buffers that you used during the graph creation
// are mapped to the device memory with the ggml_metal_add_buffer() function. This mapping is
// used during the graph evaluation to determine the arguments of the compute kernels.
//
// Synchronization between device and host memory (for example for input and output tensors)
// is done with the ggml_metal_set_tensor() and ggml_metal_get_tensor() functions.
//

#pragma once

#include <stddef.h>
#include <stdbool.h>

// max memory buffers that can be mapped to the device
#define GGML_METAL_MAX_BUFFERS 16

struct ggml_tensor;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_metal_context;

// number of command buffers to use
struct ggml_metal_context * ggml_metal_init(int n_cb);
void ggml_metal_free(struct ggml_metal_context * ctx);

// set the number of command buffers to use
void ggml_metal_set_n_cb(struct ggml_metal_context * ctx, int n_cb);

// creates a mapping between a host memory buffer and a device memory buffer
// - make sure to map all buffers used in the graph before calling ggml_metal_graph_compute
// - the mapping is used during computation to determine the arguments of the compute kernels
// - you don't need to keep the host memory buffer allocated as it is never accessed by Metal
// - max_size specifies the maximum size of a tensor and is used to create shared views such
//   that it is guaranteed that the tensor will fit in at least one of the views
//
bool ggml_metal_add_buffer(
        struct ggml_metal_context * ctx,
                       const char * name,
                             void * data,
                           size_t   size,
                           size_t   max_size);

// set data from host memory into the device
void ggml_metal_set_tensor(struct ggml_metal_context * ctx, struct ggml_tensor * t);

// get data from the device into host memory
void ggml_metal_get_tensor(struct ggml_metal_context * ctx, struct ggml_tensor * t);

// try to find operations that can be run concurrently in the graph
// you should run it again if the topology of your graph changes
void ggml_metal_graph_find_concurrency(struct ggml_metal_context * ctx, struct ggml_cgraph * gf);

// if the graph has been optimized for concurrently dispatch
bool ggml_metal_if_optimized(struct ggml_metal_context * ctx);

// same as ggml_graph_compute but uses Metal
// creates gf->n_threads command buffers in parallel
void ggml_metal_graph_compute(struct ggml_metal_context * ctx, struct ggml_cgraph * gf);

#ifdef __cplusplus
}
#endif

