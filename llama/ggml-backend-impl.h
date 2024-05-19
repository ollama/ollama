#pragma once

// ggml-backend internal header

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    //
    // Backend buffer
    //

    // buffer type
    typedef void * ggml_backend_buffer_type_context_t;

    struct ggml_backend_buffer_type_i {
        const char *          (*GGML_CALL get_name)        (ggml_backend_buffer_type_t buft);
        ggml_backend_buffer_t (*GGML_CALL alloc_buffer)    (ggml_backend_buffer_type_t buft, size_t size);
        size_t                (*GGML_CALL get_alignment)   (ggml_backend_buffer_type_t buft); // tensor alignment
        size_t                (*GGML_CALL get_max_size)    (ggml_backend_buffer_type_t buft); // allocation max size
        size_t                (*GGML_CALL get_alloc_size)  (ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor); // data size needed to allocate the tensor, including padding
        bool                  (*GGML_CALL supports_backend)(ggml_backend_buffer_type_t buft, ggml_backend_t backend); // check if the buffer type is usable by the backend
        // check if tensor data is in host memory
        // should be equivalent to supports_backend(buft, ggml_backend_cpu_init())
        bool                  (*GGML_CALL is_host)         (ggml_backend_buffer_type_t buft);
    };

    struct ggml_backend_buffer_type {
        struct ggml_backend_buffer_type_i  iface;
        ggml_backend_buffer_type_context_t context;
    };

    // buffer
    typedef void * ggml_backend_buffer_context_t;

    struct ggml_backend_buffer_i {
        const char * (*GGML_CALL get_name)   (ggml_backend_buffer_t buffer);
        void         (*GGML_CALL free_buffer)(ggml_backend_buffer_t buffer);
        void *       (*GGML_CALL get_base)   (ggml_backend_buffer_t buffer);
        void         (*GGML_CALL init_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
        void         (*GGML_CALL set_tensor) (ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void         (*GGML_CALL get_tensor) (ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        bool         (*GGML_CALL cpy_tensor) (ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst); // dst is in the buffer, src may be in any buffer
        void         (*GGML_CALL clear)      (ggml_backend_buffer_t buffer, uint8_t value);
        void         (*GGML_CALL reset)      (ggml_backend_buffer_t buffer); // reset any internal state due to tensor initialization, such as tensor extras
    };

    struct ggml_backend_buffer {
        struct ggml_backend_buffer_i  iface;
        ggml_backend_buffer_type_t    buft;
        ggml_backend_buffer_context_t context;
        size_t size;
        enum ggml_backend_buffer_usage usage;
    };

    GGML_CALL ggml_backend_buffer_t ggml_backend_buffer_init(
                   ggml_backend_buffer_type_t      buft,
            struct ggml_backend_buffer_i           iface,
                   ggml_backend_buffer_context_t   context,
                   size_t                          size);

    // do not use directly, use ggml_backend_tensor_copy instead
    bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst);

    // buffer that contains a collection of buffers
    GGML_CALL ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers);
    GGML_CALL bool                  ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer);
    GGML_CALL void                  ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage);

    //
    // Backend
    //

    typedef void * ggml_backend_context_t;

    struct ggml_backend_i {
        const char * (*GGML_CALL get_name)(ggml_backend_t backend);

        void (*GGML_CALL free)(ggml_backend_t backend);

        // buffer allocation
        ggml_backend_buffer_type_t (*GGML_CALL get_default_buffer_type)(ggml_backend_t backend);

        // (optional) asynchronous tensor data access
        void (*GGML_CALL set_tensor_async)(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*GGML_CALL get_tensor_async)(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        bool (*GGML_CALL cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

        // (optional) complete all pending operations
        void (*GGML_CALL synchronize)(ggml_backend_t backend);

        // compute graph with a plan (not used currently)
        ggml_backend_graph_plan_t (*GGML_CALL graph_plan_create) (ggml_backend_t backend, const struct ggml_cgraph * cgraph);
        void                      (*GGML_CALL graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);

        // compute graph with a plan
        enum ggml_status (*GGML_CALL graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);
        // compute graph without a plan (async)
        enum ggml_status (*GGML_CALL graph_compute)     (ggml_backend_t backend, struct ggml_cgraph * cgraph);

        // check if the backend supports an operation
        bool (*GGML_CALL supports_op)(ggml_backend_t backend, const struct ggml_tensor * op);

        // check if the backend wants to run an operation, even if the weights are allocated in a CPU buffer
        // these should be expensive operations with large batch sizes that may benefit from running on this backend
        // even if the weight has to be copied from the CPU temporarily
        bool (*GGML_CALL offload_op)(ggml_backend_t backend, const struct ggml_tensor * op);

        // (optional) event synchronization
        ggml_backend_event_t (*GGML_CALL event_new)         (ggml_backend_t backend);
        void                 (*GGML_CALL event_free)        (ggml_backend_event_t event);
        void                 (*GGML_CALL event_record)      (ggml_backend_event_t event);
        void                 (*GGML_CALL event_wait)        (ggml_backend_t backend, ggml_backend_event_t event);
        void                 (*GGML_CALL event_synchronize) (ggml_backend_event_t event);
    };

    struct ggml_backend {
        ggml_guid_t guid;

        struct ggml_backend_i iface;
        ggml_backend_context_t context;
    };

    struct ggml_backend_event {
        ggml_backend_t backend;
        void * context;
    };

    //
    // Backend registry
    //

    typedef ggml_backend_t (*GGML_CALL ggml_backend_init_fn)(const char * params, void * user_data);

    GGML_CALL void ggml_backend_register(const char * name, ggml_backend_init_fn init_fn, ggml_backend_buffer_type_t default_buffer_type, void * user_data);

#ifdef  __cplusplus
}
#endif
