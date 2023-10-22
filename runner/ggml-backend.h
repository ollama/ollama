/**
 * llama.cpp - git 465219b9143ac01db0990bbcb0a081ef72ec2008
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

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif
    struct ggml_backend;
    struct ggml_backend_buffer;

    // type-erased backend-specific types / wrappers
    typedef void * ggml_backend_context_t;
    typedef void * ggml_backend_graph_plan_t;
    typedef void * ggml_backend_buffer_context_t;

    // avoid accessing internals of these types
    typedef struct ggml_backend        * ggml_backend_t;
    typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

    //
    // backend buffer
    //

    struct ggml_backend_buffer_i {
        void   (*free_buffer)   (ggml_backend_buffer_t buffer);
        void * (*get_base)      (ggml_backend_buffer_t buffer); // get base pointer
        size_t (*get_alloc_size)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor); // pre-allocation callback
        void   (*init_tensor)   (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor); // post-allocation callback
        void   (*free_tensor)   (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor); // pre-free callback
    };

    // TODO: hide behind API
    struct ggml_backend_buffer {
        struct ggml_backend_buffer_i iface;

        ggml_backend_t                backend;
        ggml_backend_buffer_context_t context;

        size_t size;
    };

    // backend buffer functions
    GGML_API ggml_backend_buffer_t ggml_backend_buffer_init(
            struct ggml_backend                  * backend,
            struct ggml_backend_buffer_i           iface,
                   ggml_backend_buffer_context_t   context,
                   size_t                          size);

    GGML_API void   ggml_backend_buffer_free          (ggml_backend_buffer_t buffer);
    GGML_API size_t ggml_backend_buffer_get_alignment (ggml_backend_buffer_t buffer);
    GGML_API void * ggml_backend_buffer_get_base      (ggml_backend_buffer_t buffer);
    GGML_API size_t ggml_backend_buffer_get_size      (ggml_backend_buffer_t buffer);
    GGML_API size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
    GGML_API void   ggml_backend_buffer_init_tensor   (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
    GGML_API void   ggml_backend_buffer_free_tensor   (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);

    //
    // backend
    //

    struct ggml_backend_i {
        const char * (*get_name)(ggml_backend_t backend);

        void (*free)(ggml_backend_t backend);

        // buffer allocation
        ggml_backend_buffer_t (*alloc_buffer)(ggml_backend_t backend, size_t size);

        // get buffer alignment
        size_t (*get_alignment)(ggml_backend_t backend);

        // tensor data access
        // these functions can be asynchronous, helper functions are provided for synchronous access that automatically call synchronize
        void (*set_tensor_async)(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*get_tensor_async)(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        void (*synchronize)     (ggml_backend_t backend);

        // (optional) copy tensor between different backends, allow for single-copy tranfers
        void (*cpy_tensor_from)(ggml_backend_t backend, struct ggml_tensor * src, struct ggml_tensor * dst);
        void (*cpy_tensor_to)  (ggml_backend_t backend, struct ggml_tensor * src, struct ggml_tensor * dst);

        // compute graph with a plan
        ggml_backend_graph_plan_t (*graph_plan_create) (ggml_backend_t backend, struct ggml_cgraph * cgraph);
        void                      (*graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
        void                      (*graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);

        // compute graph without a plan
        void (*graph_compute)(ggml_backend_t backend, struct ggml_cgraph * cgraph);

        // check if the backend supports an operation
        bool (*supports_op)(ggml_backend_t backend, const struct ggml_tensor * op);
    };

    // TODO: hide behind API
    struct ggml_backend {
        struct ggml_backend_i iface;

        ggml_backend_context_t context;
    };

    // backend helper functions
    GGML_API ggml_backend_t ggml_get_backend(const struct ggml_tensor * tensor);

    GGML_API const char * ggml_backend_name(ggml_backend_t backend);
    GGML_API void         ggml_backend_free(ggml_backend_t backend);

    GGML_API ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size);

    GGML_API size_t ggml_backend_get_alignment(ggml_backend_t backend);

    GGML_API void ggml_backend_tensor_set_async(      struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    GGML_API void ggml_backend_tensor_get_async(const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);

    GGML_API void ggml_backend_tensor_set(      struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    GGML_API void ggml_backend_tensor_get(const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);

    GGML_API void ggml_backend_synchronize(ggml_backend_t backend);

    GGML_API ggml_backend_graph_plan_t ggml_backend_graph_plan_create (ggml_backend_t backend, struct ggml_cgraph * cgraph);

    GGML_API void ggml_backend_graph_plan_free   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
    GGML_API void ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan);
    GGML_API void ggml_backend_graph_compute     (ggml_backend_t backend, struct ggml_cgraph * cgraph);
    GGML_API bool ggml_backend_supports_op       (ggml_backend_t backend, const struct ggml_tensor * op);

    // tensor copy between different backends
    GGML_API void ggml_backend_tensor_copy(struct ggml_tensor * src, struct ggml_tensor * dst);

    //
    // CPU backend
    //

    GGML_API ggml_backend_t ggml_backend_cpu_init(void);

    GGML_API bool ggml_backend_is_cpu(ggml_backend_t backend);

    GGML_API void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads);

    GGML_API ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(ggml_backend_t backend_cpu, void * ptr, size_t size);

#ifdef  __cplusplus
}
#endif
