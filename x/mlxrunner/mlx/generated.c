// This code is auto-generated; DO NOT EDIT.

#include "generated.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

size_t (*mlx_dtype_size_)(mlx_dtype dtype) = NULL;
int (*mlx_array_tostring_)(mlx_string* str, const mlx_array arr) = NULL;
mlx_array (*mlx_array_new_)(void) = NULL;
int (*mlx_array_free_)(mlx_array arr) = NULL;
mlx_array (*mlx_array_new_bool_)(bool val) = NULL;
mlx_array (*mlx_array_new_int_)(int val) = NULL;
mlx_array (*mlx_array_new_float32_)(float val) = NULL;
mlx_array (*mlx_array_new_float_)(float val) = NULL;
mlx_array (*mlx_array_new_float64_)(double val) = NULL;
mlx_array (*mlx_array_new_double_)(double val) = NULL;
mlx_array (*mlx_array_new_complex_)(float real_val, float imag_val) = NULL;
mlx_array (*mlx_array_new_data_)(
    const void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype) = NULL;
int (*mlx_array_set_)(mlx_array* arr, const mlx_array src) = NULL;
int (*mlx_array_set_bool_)(mlx_array* arr, bool val) = NULL;
int (*mlx_array_set_int_)(mlx_array* arr, int val) = NULL;
int (*mlx_array_set_float32_)(mlx_array* arr, float val) = NULL;
int (*mlx_array_set_float_)(mlx_array* arr, float val) = NULL;
int (*mlx_array_set_float64_)(mlx_array* arr, double val) = NULL;
int (*mlx_array_set_double_)(mlx_array* arr, double val) = NULL;
int (*mlx_array_set_complex_)(mlx_array* arr, float real_val, float imag_val) = NULL;
int (*mlx_array_set_data_)(
    mlx_array* arr,
    const void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype) = NULL;
size_t (*mlx_array_itemsize_)(const mlx_array arr) = NULL;
size_t (*mlx_array_size_)(const mlx_array arr) = NULL;
size_t (*mlx_array_nbytes_)(const mlx_array arr) = NULL;
size_t (*mlx_array_ndim_)(const mlx_array arr) = NULL;
const int * (*mlx_array_shape_)(const mlx_array arr) = NULL;
const size_t * (*mlx_array_strides_)(const mlx_array arr) = NULL;
int (*mlx_array_dim_)(const mlx_array arr, int dim) = NULL;
mlx_dtype (*mlx_array_dtype_)(const mlx_array arr) = NULL;
int (*mlx_array_eval_)(mlx_array arr) = NULL;
int (*mlx_array_item_bool_)(bool* res, const mlx_array arr) = NULL;
int (*mlx_array_item_uint8_)(uint8_t* res, const mlx_array arr) = NULL;
int (*mlx_array_item_uint16_)(uint16_t* res, const mlx_array arr) = NULL;
int (*mlx_array_item_uint32_)(uint32_t* res, const mlx_array arr) = NULL;
int (*mlx_array_item_uint64_)(uint64_t* res, const mlx_array arr) = NULL;
int (*mlx_array_item_int8_)(int8_t* res, const mlx_array arr) = NULL;
int (*mlx_array_item_int16_)(int16_t* res, const mlx_array arr) = NULL;
int (*mlx_array_item_int32_)(int32_t* res, const mlx_array arr) = NULL;
int (*mlx_array_item_int64_)(int64_t* res, const mlx_array arr) = NULL;
int (*mlx_array_item_float32_)(float* res, const mlx_array arr) = NULL;
int (*mlx_array_item_float64_)(double* res, const mlx_array arr) = NULL;
int (*mlx_array_item_complex64_)(float _Complex* res, const mlx_array arr) = NULL;
int (*mlx_array_item_float16_)(float16_t* res, const mlx_array arr) = NULL;
int (*mlx_array_item_bfloat16_)(bfloat16_t* res, const mlx_array arr) = NULL;
const bool * (*mlx_array_data_bool_)(const mlx_array arr) = NULL;
const uint8_t * (*mlx_array_data_uint8_)(const mlx_array arr) = NULL;
const uint16_t * (*mlx_array_data_uint16_)(const mlx_array arr) = NULL;
const uint32_t * (*mlx_array_data_uint32_)(const mlx_array arr) = NULL;
const uint64_t * (*mlx_array_data_uint64_)(const mlx_array arr) = NULL;
const int8_t * (*mlx_array_data_int8_)(const mlx_array arr) = NULL;
const int16_t * (*mlx_array_data_int16_)(const mlx_array arr) = NULL;
const int32_t * (*mlx_array_data_int32_)(const mlx_array arr) = NULL;
const int64_t * (*mlx_array_data_int64_)(const mlx_array arr) = NULL;
const float * (*mlx_array_data_float32_)(const mlx_array arr) = NULL;
const double * (*mlx_array_data_float64_)(const mlx_array arr) = NULL;
const float _Complex * (*mlx_array_data_complex64_)(const mlx_array arr) = NULL;
const float16_t * (*mlx_array_data_float16_)(const mlx_array arr) = NULL;
const bfloat16_t * (*mlx_array_data_bfloat16_)(const mlx_array arr) = NULL;
int (*_mlx_array_is_available_)(bool* res, const mlx_array arr) = NULL;
int (*_mlx_array_wait_)(const mlx_array arr) = NULL;
int (*_mlx_array_is_contiguous_)(bool* res, const mlx_array arr) = NULL;
int (*_mlx_array_is_row_contiguous_)(bool* res, const mlx_array arr) = NULL;
int (*_mlx_array_is_col_contiguous_)(bool* res, const mlx_array arr) = NULL;
mlx_closure (*mlx_closure_new_)(void) = NULL;
int (*mlx_closure_free_)(mlx_closure cls) = NULL;
mlx_closure (*mlx_closure_new_func_)(
    int (*fun)(mlx_vector_array*, const mlx_vector_array)) = NULL;
mlx_closure (*mlx_closure_new_func_payload_)(
    int (*fun)(mlx_vector_array*, const mlx_vector_array, void*),
    void* payload,
    void (*dtor)(void*)) = NULL;
int (*mlx_closure_set_)(mlx_closure* cls, const mlx_closure src) = NULL;
int (*mlx_closure_apply_)(
    mlx_vector_array* res,
    mlx_closure cls,
    const mlx_vector_array input) = NULL;
mlx_closure (*mlx_closure_new_unary_)(int (*fun)(mlx_array*, const mlx_array)) = NULL;
mlx_closure_kwargs (*mlx_closure_kwargs_new_)(void) = NULL;
int (*mlx_closure_kwargs_free_)(mlx_closure_kwargs cls) = NULL;
mlx_closure_kwargs (*mlx_closure_kwargs_new_func_)(int (*fun)(
    mlx_vector_array*,
    const mlx_vector_array,
    const mlx_map_string_to_array)) = NULL;
mlx_closure_kwargs (*mlx_closure_kwargs_new_func_payload_)(
    int (*fun)(
        mlx_vector_array*,
        const mlx_vector_array,
        const mlx_map_string_to_array,
        void*),
    void* payload,
    void (*dtor)(void*)) = NULL;
int (*mlx_closure_kwargs_set_)(
    mlx_closure_kwargs* cls,
    const mlx_closure_kwargs src) = NULL;
int (*mlx_closure_kwargs_apply_)(
    mlx_vector_array* res,
    mlx_closure_kwargs cls,
    const mlx_vector_array input_0,
    const mlx_map_string_to_array input_1) = NULL;
mlx_closure_value_and_grad (*mlx_closure_value_and_grad_new_)(void) = NULL;
int (*mlx_closure_value_and_grad_free_)(mlx_closure_value_and_grad cls) = NULL;
mlx_closure_value_and_grad (*mlx_closure_value_and_grad_new_func_)(
    int (*fun)(mlx_vector_array*, mlx_vector_array*, const mlx_vector_array)) = NULL;
mlx_closure_value_and_grad (*mlx_closure_value_and_grad_new_func_payload_)(
    int (*fun)(
        mlx_vector_array*,
        mlx_vector_array*,
        const mlx_vector_array,
        void*),
    void* payload,
    void (*dtor)(void*)) = NULL;
int (*mlx_closure_value_and_grad_set_)(
    mlx_closure_value_and_grad* cls,
    const mlx_closure_value_and_grad src) = NULL;
int (*mlx_closure_value_and_grad_apply_)(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    mlx_closure_value_and_grad cls,
    const mlx_vector_array input) = NULL;
mlx_closure_custom (*mlx_closure_custom_new_)(void) = NULL;
int (*mlx_closure_custom_free_)(mlx_closure_custom cls) = NULL;
mlx_closure_custom (*mlx_closure_custom_new_func_)(int (*fun)(
    mlx_vector_array*,
    const mlx_vector_array,
    const mlx_vector_array,
    const mlx_vector_array)) = NULL;
mlx_closure_custom (*mlx_closure_custom_new_func_payload_)(
    int (*fun)(
        mlx_vector_array*,
        const mlx_vector_array,
        const mlx_vector_array,
        const mlx_vector_array,
        void*),
    void* payload,
    void (*dtor)(void*)) = NULL;
int (*mlx_closure_custom_set_)(
    mlx_closure_custom* cls,
    const mlx_closure_custom src) = NULL;
int (*mlx_closure_custom_apply_)(
    mlx_vector_array* res,
    mlx_closure_custom cls,
    const mlx_vector_array input_0,
    const mlx_vector_array input_1,
    const mlx_vector_array input_2) = NULL;
mlx_closure_custom_jvp (*mlx_closure_custom_jvp_new_)(void) = NULL;
int (*mlx_closure_custom_jvp_free_)(mlx_closure_custom_jvp cls) = NULL;
mlx_closure_custom_jvp (*mlx_closure_custom_jvp_new_func_)(int (*fun)(
    mlx_vector_array*,
    const mlx_vector_array,
    const mlx_vector_array,
    const int*,
    size_t _num)) = NULL;
mlx_closure_custom_jvp (*mlx_closure_custom_jvp_new_func_payload_)(
    int (*fun)(
        mlx_vector_array*,
        const mlx_vector_array,
        const mlx_vector_array,
        const int*,
        size_t _num,
        void*),
    void* payload,
    void (*dtor)(void*)) = NULL;
int (*mlx_closure_custom_jvp_set_)(
    mlx_closure_custom_jvp* cls,
    const mlx_closure_custom_jvp src) = NULL;
int (*mlx_closure_custom_jvp_apply_)(
    mlx_vector_array* res,
    mlx_closure_custom_jvp cls,
    const mlx_vector_array input_0,
    const mlx_vector_array input_1,
    const int* input_2,
    size_t input_2_num) = NULL;
mlx_closure_custom_vmap (*mlx_closure_custom_vmap_new_)(void) = NULL;
int (*mlx_closure_custom_vmap_free_)(mlx_closure_custom_vmap cls) = NULL;
mlx_closure_custom_vmap (*mlx_closure_custom_vmap_new_func_)(int (*fun)(
    mlx_vector_array*,
    mlx_vector_int*,
    const mlx_vector_array,
    const int*,
    size_t _num)) = NULL;
mlx_closure_custom_vmap (*mlx_closure_custom_vmap_new_func_payload_)(
    int (*fun)(
        mlx_vector_array*,
        mlx_vector_int*,
        const mlx_vector_array,
        const int*,
        size_t _num,
        void*),
    void* payload,
    void (*dtor)(void*)) = NULL;
int (*mlx_closure_custom_vmap_set_)(
    mlx_closure_custom_vmap* cls,
    const mlx_closure_custom_vmap src) = NULL;
int (*mlx_closure_custom_vmap_apply_)(
    mlx_vector_array* res_0,
    mlx_vector_int* res_1,
    mlx_closure_custom_vmap cls,
    const mlx_vector_array input_0,
    const int* input_1,
    size_t input_1_num) = NULL;
int (*mlx_compile_)(mlx_closure* res, const mlx_closure fun, bool shapeless) = NULL;
int (*mlx_detail_compile_)(
    mlx_closure* res,
    const mlx_closure fun,
    uintptr_t fun_id,
    bool shapeless,
    const uint64_t* constants,
    size_t constants_num) = NULL;
int (*mlx_detail_compile_clear_cache_)(void) = NULL;
int (*mlx_detail_compile_erase_)(uintptr_t fun_id) = NULL;
int (*mlx_disable_compile_)(void) = NULL;
int (*mlx_enable_compile_)(void) = NULL;
int (*mlx_set_compile_mode_)(mlx_compile_mode mode) = NULL;
mlx_device (*mlx_device_new_)(void) = NULL;
mlx_device (*mlx_device_new_type_)(mlx_device_type type, int index) = NULL;
int (*mlx_device_free_)(mlx_device dev) = NULL;
int (*mlx_device_set_)(mlx_device* dev, const mlx_device src) = NULL;
int (*mlx_device_tostring_)(mlx_string* str, mlx_device dev) = NULL;
bool (*mlx_device_equal_)(mlx_device lhs, mlx_device rhs) = NULL;
int (*mlx_device_get_index_)(int* index, mlx_device dev) = NULL;
int (*mlx_device_get_type_)(mlx_device_type* type, mlx_device dev) = NULL;
int (*mlx_get_default_device_)(mlx_device* dev) = NULL;
int (*mlx_set_default_device_)(mlx_device dev) = NULL;
int (*mlx_distributed_group_rank_)(mlx_distributed_group group) = NULL;
int (*mlx_distributed_group_size_)(mlx_distributed_group group) = NULL;
mlx_distributed_group (*mlx_distributed_group_split_)(mlx_distributed_group group, int color, int key) = NULL;
bool (*mlx_distributed_is_available_)(void) = NULL;
mlx_distributed_group (*mlx_distributed_init_)(bool strict) = NULL;
int (*mlx_distributed_all_gather_)(
    mlx_array* res,
    const mlx_array x,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream S) = NULL;
int (*mlx_distributed_all_max_)(
    mlx_array* res,
    const mlx_array x,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s) = NULL;
int (*mlx_distributed_all_min_)(
    mlx_array* res,
    const mlx_array x,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s) = NULL;
int (*mlx_distributed_all_sum_)(
    mlx_array* res,
    const mlx_array x,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s) = NULL;
int (*mlx_distributed_recv_)(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    int src,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s) = NULL;
int (*mlx_distributed_recv_like_)(
    mlx_array* res,
    const mlx_array x,
    int src,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s) = NULL;
int (*mlx_distributed_send_)(
    mlx_array* res,
    const mlx_array x,
    int dst,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s) = NULL;
int (*mlx_distributed_sum_scatter_)(
    mlx_array* res,
    const mlx_array x,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s) = NULL;
void (*mlx_set_error_handler_)(
    mlx_error_handler_func handler,
    void* data,
    void (*dtor)(void*)) = NULL;
void (*_mlx_error_)(const char* file, const int line, const char* fmt, ...) = NULL;
int (*mlx_export_function_)(
    const char* file,
    const mlx_closure fun,
    const mlx_vector_array args,
    bool shapeless) = NULL;
int (*mlx_export_function_kwargs_)(
    const char* file,
    const mlx_closure_kwargs fun,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs,
    bool shapeless) = NULL;
mlx_function_exporter (*mlx_function_exporter_new_)(
    const char* file,
    const mlx_closure fun,
    bool shapeless) = NULL;
int (*mlx_function_exporter_free_)(mlx_function_exporter xfunc) = NULL;
int (*mlx_function_exporter_apply_)(
    const mlx_function_exporter xfunc,
    const mlx_vector_array args) = NULL;
int (*mlx_function_exporter_apply_kwargs_)(
    const mlx_function_exporter xfunc,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs) = NULL;
mlx_imported_function (*mlx_imported_function_new_)(const char* file) = NULL;
int (*mlx_imported_function_free_)(mlx_imported_function xfunc) = NULL;
int (*mlx_imported_function_apply_)(
    mlx_vector_array* res,
    const mlx_imported_function xfunc,
    const mlx_vector_array args) = NULL;
int (*mlx_imported_function_apply_kwargs_)(
    mlx_vector_array* res,
    const mlx_imported_function xfunc,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs) = NULL;
mlx_fast_cuda_kernel_config (*mlx_fast_cuda_kernel_config_new_)(void) = NULL;
void (*mlx_fast_cuda_kernel_config_free_)(mlx_fast_cuda_kernel_config cls) = NULL;
int (*mlx_fast_cuda_kernel_config_add_output_arg_)(
    mlx_fast_cuda_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype) = NULL;
int (*mlx_fast_cuda_kernel_config_set_grid_)(
    mlx_fast_cuda_kernel_config cls,
    int grid1,
    int grid2,
    int grid3) = NULL;
int (*mlx_fast_cuda_kernel_config_set_thread_group_)(
    mlx_fast_cuda_kernel_config cls,
    int thread1,
    int thread2,
    int thread3) = NULL;
int (*mlx_fast_cuda_kernel_config_set_init_value_)(
    mlx_fast_cuda_kernel_config cls,
    float value) = NULL;
int (*mlx_fast_cuda_kernel_config_set_verbose_)(
    mlx_fast_cuda_kernel_config cls,
    bool verbose) = NULL;
int (*mlx_fast_cuda_kernel_config_add_template_arg_dtype_)(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    mlx_dtype dtype) = NULL;
int (*mlx_fast_cuda_kernel_config_add_template_arg_int_)(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    int value) = NULL;
int (*mlx_fast_cuda_kernel_config_add_template_arg_bool_)(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    bool value) = NULL;
mlx_fast_cuda_kernel (*mlx_fast_cuda_kernel_new_)(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    int shared_memory) = NULL;
void (*mlx_fast_cuda_kernel_free_)(mlx_fast_cuda_kernel cls) = NULL;
int (*mlx_fast_cuda_kernel_apply_)(
    mlx_vector_array* outputs,
    mlx_fast_cuda_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_cuda_kernel_config config,
    const mlx_stream stream) = NULL;
int (*mlx_fast_layer_norm_)(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    const mlx_array bias /* may be null */,
    float eps,
    const mlx_stream s) = NULL;
mlx_fast_metal_kernel_config (*mlx_fast_metal_kernel_config_new_)(void) = NULL;
void (*mlx_fast_metal_kernel_config_free_)(mlx_fast_metal_kernel_config cls) = NULL;
int (*mlx_fast_metal_kernel_config_add_output_arg_)(
    mlx_fast_metal_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype) = NULL;
int (*mlx_fast_metal_kernel_config_set_grid_)(
    mlx_fast_metal_kernel_config cls,
    int grid1,
    int grid2,
    int grid3) = NULL;
int (*mlx_fast_metal_kernel_config_set_thread_group_)(
    mlx_fast_metal_kernel_config cls,
    int thread1,
    int thread2,
    int thread3) = NULL;
int (*mlx_fast_metal_kernel_config_set_init_value_)(
    mlx_fast_metal_kernel_config cls,
    float value) = NULL;
int (*mlx_fast_metal_kernel_config_set_verbose_)(
    mlx_fast_metal_kernel_config cls,
    bool verbose) = NULL;
int (*mlx_fast_metal_kernel_config_add_template_arg_dtype_)(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    mlx_dtype dtype) = NULL;
int (*mlx_fast_metal_kernel_config_add_template_arg_int_)(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    int value) = NULL;
int (*mlx_fast_metal_kernel_config_add_template_arg_bool_)(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    bool value) = NULL;
mlx_fast_metal_kernel (*mlx_fast_metal_kernel_new_)(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    bool atomic_outputs) = NULL;
void (*mlx_fast_metal_kernel_free_)(mlx_fast_metal_kernel cls) = NULL;
int (*mlx_fast_metal_kernel_apply_)(
    mlx_vector_array* outputs,
    mlx_fast_metal_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_metal_kernel_config config,
    const mlx_stream stream) = NULL;
int (*mlx_fast_rms_norm_)(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    float eps,
    const mlx_stream s) = NULL;
int (*mlx_fast_rope_)(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    int offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s) = NULL;
int (*mlx_fast_scaled_dot_product_attention_)(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array keys,
    const mlx_array values,
    float scale,
    const char* mask_mode,
    const mlx_array mask_arr /* may be null */,
    const mlx_array sinks /* may be null */,
    const mlx_stream s) = NULL;
int (*mlx_fft_fft_)(
    mlx_array* res,
    const mlx_array a,
    int n,
    int axis,
    const mlx_stream s) = NULL;
int (*mlx_fft_fft2_)(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) = NULL;
int (*mlx_fft_fftn_)(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) = NULL;
int (*mlx_fft_fftshift_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) = NULL;
int (*mlx_fft_ifft_)(
    mlx_array* res,
    const mlx_array a,
    int n,
    int axis,
    const mlx_stream s) = NULL;
int (*mlx_fft_ifft2_)(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) = NULL;
int (*mlx_fft_ifftn_)(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) = NULL;
int (*mlx_fft_ifftshift_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) = NULL;
int (*mlx_fft_irfft_)(
    mlx_array* res,
    const mlx_array a,
    int n,
    int axis,
    const mlx_stream s) = NULL;
int (*mlx_fft_irfft2_)(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) = NULL;
int (*mlx_fft_irfftn_)(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) = NULL;
int (*mlx_fft_rfft_)(
    mlx_array* res,
    const mlx_array a,
    int n,
    int axis,
    const mlx_stream s) = NULL;
int (*mlx_fft_rfft2_)(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) = NULL;
int (*mlx_fft_rfftn_)(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) = NULL;
mlx_io_reader (*mlx_io_reader_new_)(void* desc, mlx_io_vtable vtable) = NULL;
int (*mlx_io_reader_descriptor_)(void** desc_, mlx_io_reader io) = NULL;
int (*mlx_io_reader_tostring_)(mlx_string* str_, mlx_io_reader io) = NULL;
int (*mlx_io_reader_free_)(mlx_io_reader io) = NULL;
mlx_io_writer (*mlx_io_writer_new_)(void* desc, mlx_io_vtable vtable) = NULL;
int (*mlx_io_writer_descriptor_)(void** desc_, mlx_io_writer io) = NULL;
int (*mlx_io_writer_tostring_)(mlx_string* str_, mlx_io_writer io) = NULL;
int (*mlx_io_writer_free_)(mlx_io_writer io) = NULL;
int (*mlx_load_reader_)(
    mlx_array* res,
    mlx_io_reader in_stream,
    const mlx_stream s) = NULL;
int (*mlx_load_)(mlx_array* res, const char* file, const mlx_stream s) = NULL;
int (*mlx_load_safetensors_reader_)(
    mlx_map_string_to_array* res_0,
    mlx_map_string_to_string* res_1,
    mlx_io_reader in_stream,
    const mlx_stream s) = NULL;
int (*mlx_load_safetensors_)(
    mlx_map_string_to_array* res_0,
    mlx_map_string_to_string* res_1,
    const char* file,
    const mlx_stream s) = NULL;
int (*mlx_save_writer_)(mlx_io_writer out_stream, const mlx_array a) = NULL;
int (*mlx_save_)(const char* file, const mlx_array a) = NULL;
int (*mlx_save_safetensors_writer_)(
    mlx_io_writer in_stream,
    const mlx_map_string_to_array param,
    const mlx_map_string_to_string metadata) = NULL;
int (*mlx_save_safetensors_)(
    const char* file,
    const mlx_map_string_to_array param,
    const mlx_map_string_to_string metadata) = NULL;
int (*mlx_linalg_cholesky_)(
    mlx_array* res,
    const mlx_array a,
    bool upper,
    const mlx_stream s) = NULL;
int (*mlx_linalg_cholesky_inv_)(
    mlx_array* res,
    const mlx_array a,
    bool upper,
    const mlx_stream s) = NULL;
int (*mlx_linalg_cross_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    int axis,
    const mlx_stream s) = NULL;
int (*mlx_linalg_eig_)(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const mlx_stream s) = NULL;
int (*mlx_linalg_eigh_)(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const char* UPLO,
    const mlx_stream s) = NULL;
int (*mlx_linalg_eigvals_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_linalg_eigvalsh_)(
    mlx_array* res,
    const mlx_array a,
    const char* UPLO,
    const mlx_stream s) = NULL;
int (*mlx_linalg_inv_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_linalg_lu_)(mlx_vector_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_linalg_lu_factor_)(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const mlx_stream s) = NULL;
int (*mlx_linalg_norm_)(
    mlx_array* res,
    const mlx_array a,
    double ord,
    const int* axis /* may be null */,
    size_t axis_num,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_linalg_norm_matrix_)(
    mlx_array* res,
    const mlx_array a,
    const char* ord,
    const int* axis /* may be null */,
    size_t axis_num,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_linalg_norm_l2_)(
    mlx_array* res,
    const mlx_array a,
    const int* axis /* may be null */,
    size_t axis_num,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_linalg_pinv_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_linalg_qr_)(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const mlx_stream s) = NULL;
int (*mlx_linalg_solve_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_linalg_solve_triangular_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    bool upper,
    const mlx_stream s) = NULL;
int (*mlx_linalg_svd_)(
    mlx_vector_array* res,
    const mlx_array a,
    bool compute_uv,
    const mlx_stream s) = NULL;
int (*mlx_linalg_tri_inv_)(
    mlx_array* res,
    const mlx_array a,
    bool upper,
    const mlx_stream s) = NULL;
mlx_map_string_to_array (*mlx_map_string_to_array_new_)(void) = NULL;
int (*mlx_map_string_to_array_set_)(
    mlx_map_string_to_array* map,
    const mlx_map_string_to_array src) = NULL;
int (*mlx_map_string_to_array_free_)(mlx_map_string_to_array map) = NULL;
int (*mlx_map_string_to_array_insert_)(
    mlx_map_string_to_array map,
    const char* key,
    const mlx_array value) = NULL;
int (*mlx_map_string_to_array_get_)(
    mlx_array* value,
    const mlx_map_string_to_array map,
    const char* key) = NULL;
mlx_map_string_to_array_iterator (*mlx_map_string_to_array_iterator_new_)(
    mlx_map_string_to_array map) = NULL;
int (*mlx_map_string_to_array_iterator_free_)(mlx_map_string_to_array_iterator it) = NULL;
int (*mlx_map_string_to_array_iterator_next_)(
    const char** key,
    mlx_array* value,
    mlx_map_string_to_array_iterator it) = NULL;
mlx_map_string_to_string (*mlx_map_string_to_string_new_)(void) = NULL;
int (*mlx_map_string_to_string_set_)(
    mlx_map_string_to_string* map,
    const mlx_map_string_to_string src) = NULL;
int (*mlx_map_string_to_string_free_)(mlx_map_string_to_string map) = NULL;
int (*mlx_map_string_to_string_insert_)(
    mlx_map_string_to_string map,
    const char* key,
    const char* value) = NULL;
int (*mlx_map_string_to_string_get_)(
    const char** value,
    const mlx_map_string_to_string map,
    const char* key) = NULL;
mlx_map_string_to_string_iterator (*mlx_map_string_to_string_iterator_new_)(
    mlx_map_string_to_string map) = NULL;
int (*mlx_map_string_to_string_iterator_free_)(
    mlx_map_string_to_string_iterator it) = NULL;
int (*mlx_map_string_to_string_iterator_next_)(
    const char** key,
    const char** value,
    mlx_map_string_to_string_iterator it) = NULL;
int (*mlx_clear_cache_)(void) = NULL;
int (*mlx_get_active_memory_)(size_t* res) = NULL;
int (*mlx_get_cache_memory_)(size_t* res) = NULL;
int (*mlx_get_memory_limit_)(size_t* res) = NULL;
int (*mlx_get_peak_memory_)(size_t* res) = NULL;
int (*mlx_reset_peak_memory_)(void) = NULL;
int (*mlx_set_cache_limit_)(size_t* res, size_t limit) = NULL;
int (*mlx_set_memory_limit_)(size_t* res, size_t limit) = NULL;
int (*mlx_set_wired_limit_)(size_t* res, size_t limit) = NULL;
mlx_metal_device_info_t (*mlx_metal_device_info_)(void) = NULL;
int (*mlx_metal_is_available_)(bool* res) = NULL;
int (*mlx_metal_start_capture_)(const char* path) = NULL;
int (*mlx_metal_stop_capture_)(void) = NULL;
int (*mlx_abs_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_add_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_addmm_)(
    mlx_array* res,
    const mlx_array c,
    const mlx_array a,
    const mlx_array b,
    float alpha,
    float beta,
    const mlx_stream s) = NULL;
int (*mlx_all_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_all_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_all_)(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_allclose_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    double rtol,
    double atol,
    bool equal_nan,
    const mlx_stream s) = NULL;
int (*mlx_any_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_any_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_any_)(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_arange_)(
    mlx_array* res,
    double start,
    double stop,
    double step,
    mlx_dtype dtype,
    const mlx_stream s) = NULL;
int (*mlx_arccos_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_arccosh_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_arcsin_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_arcsinh_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_arctan_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_arctan2_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_arctanh_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_argmax_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_argmax_)(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_argmin_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_argmin_)(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_argpartition_axis_)(
    mlx_array* res,
    const mlx_array a,
    int kth,
    int axis,
    const mlx_stream s) = NULL;
int (*mlx_argpartition_)(
    mlx_array* res,
    const mlx_array a,
    int kth,
    const mlx_stream s) = NULL;
int (*mlx_argsort_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    const mlx_stream s) = NULL;
int (*mlx_argsort_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_array_equal_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    bool equal_nan,
    const mlx_stream s) = NULL;
int (*mlx_as_strided_)(
    mlx_array* res,
    const mlx_array a,
    const int* shape,
    size_t shape_num,
    const int64_t* strides,
    size_t strides_num,
    size_t offset,
    const mlx_stream s) = NULL;
int (*mlx_astype_)(
    mlx_array* res,
    const mlx_array a,
    mlx_dtype dtype,
    const mlx_stream s) = NULL;
int (*mlx_atleast_1d_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_atleast_2d_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_atleast_3d_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_bitwise_and_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_bitwise_invert_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_bitwise_or_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_bitwise_xor_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_block_masked_mm_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    int block_size,
    const mlx_array mask_out /* may be null */,
    const mlx_array mask_lhs /* may be null */,
    const mlx_array mask_rhs /* may be null */,
    const mlx_stream s) = NULL;
int (*mlx_broadcast_arrays_)(
    mlx_vector_array* res,
    const mlx_vector_array inputs,
    const mlx_stream s) = NULL;
int (*mlx_broadcast_to_)(
    mlx_array* res,
    const mlx_array a,
    const int* shape,
    size_t shape_num,
    const mlx_stream s) = NULL;
int (*mlx_ceil_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_clip_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array a_min /* may be null */,
    const mlx_array a_max /* may be null */,
    const mlx_stream s) = NULL;
int (*mlx_concatenate_axis_)(
    mlx_array* res,
    const mlx_vector_array arrays,
    int axis,
    const mlx_stream s) = NULL;
int (*mlx_concatenate_)(
    mlx_array* res,
    const mlx_vector_array arrays,
    const mlx_stream s) = NULL;
int (*mlx_conjugate_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_contiguous_)(
    mlx_array* res,
    const mlx_array a,
    bool allow_col_major,
    const mlx_stream s) = NULL;
int (*mlx_conv1d_)(
    mlx_array* res,
    const mlx_array input,
    const mlx_array weight,
    int stride,
    int padding,
    int dilation,
    int groups,
    const mlx_stream s) = NULL;
int (*mlx_conv2d_)(
    mlx_array* res,
    const mlx_array input,
    const mlx_array weight,
    int stride_0,
    int stride_1,
    int padding_0,
    int padding_1,
    int dilation_0,
    int dilation_1,
    int groups,
    const mlx_stream s) = NULL;
int (*mlx_conv3d_)(
    mlx_array* res,
    const mlx_array input,
    const mlx_array weight,
    int stride_0,
    int stride_1,
    int stride_2,
    int padding_0,
    int padding_1,
    int padding_2,
    int dilation_0,
    int dilation_1,
    int dilation_2,
    int groups,
    const mlx_stream s) = NULL;
int (*mlx_conv_general_)(
    mlx_array* res,
    const mlx_array input,
    const mlx_array weight,
    const int* stride,
    size_t stride_num,
    const int* padding_lo,
    size_t padding_lo_num,
    const int* padding_hi,
    size_t padding_hi_num,
    const int* kernel_dilation,
    size_t kernel_dilation_num,
    const int* input_dilation,
    size_t input_dilation_num,
    int groups,
    bool flip,
    const mlx_stream s) = NULL;
int (*mlx_conv_transpose1d_)(
    mlx_array* res,
    const mlx_array input,
    const mlx_array weight,
    int stride,
    int padding,
    int dilation,
    int output_padding,
    int groups,
    const mlx_stream s) = NULL;
int (*mlx_conv_transpose2d_)(
    mlx_array* res,
    const mlx_array input,
    const mlx_array weight,
    int stride_0,
    int stride_1,
    int padding_0,
    int padding_1,
    int dilation_0,
    int dilation_1,
    int output_padding_0,
    int output_padding_1,
    int groups,
    const mlx_stream s) = NULL;
int (*mlx_conv_transpose3d_)(
    mlx_array* res,
    const mlx_array input,
    const mlx_array weight,
    int stride_0,
    int stride_1,
    int stride_2,
    int padding_0,
    int padding_1,
    int padding_2,
    int dilation_0,
    int dilation_1,
    int dilation_2,
    int output_padding_0,
    int output_padding_1,
    int output_padding_2,
    int groups,
    const mlx_stream s) = NULL;
int (*mlx_copy_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_cos_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_cosh_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_cummax_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool reverse,
    bool inclusive,
    const mlx_stream s) = NULL;
int (*mlx_cummin_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool reverse,
    bool inclusive,
    const mlx_stream s) = NULL;
int (*mlx_cumprod_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool reverse,
    bool inclusive,
    const mlx_stream s) = NULL;
int (*mlx_cumsum_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool reverse,
    bool inclusive,
    const mlx_stream s) = NULL;
int (*mlx_degrees_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_depends_)(
    mlx_vector_array* res,
    const mlx_vector_array inputs,
    const mlx_vector_array dependencies) = NULL;
int (*mlx_dequantize_)(
    mlx_array* res,
    const mlx_array w,
    const mlx_array scales,
    const mlx_array biases /* may be null */,
    mlx_optional_int group_size,
    mlx_optional_int bits,
    const char* mode,
    mlx_optional_dtype dtype,
    const mlx_stream s) = NULL;
int (*mlx_diag_)(mlx_array* res, const mlx_array a, int k, const mlx_stream s) = NULL;
int (*mlx_diagonal_)(
    mlx_array* res,
    const mlx_array a,
    int offset,
    int axis1,
    int axis2,
    const mlx_stream s) = NULL;
int (*mlx_divide_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_divmod_)(
    mlx_vector_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_einsum_)(
    mlx_array* res,
    const char* subscripts,
    const mlx_vector_array operands,
    const mlx_stream s) = NULL;
int (*mlx_equal_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_erf_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_erfinv_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_exp_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_expand_dims_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) = NULL;
int (*mlx_expand_dims_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    const mlx_stream s) = NULL;
int (*mlx_expm1_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_eye_)(
    mlx_array* res,
    int n,
    int m,
    int k,
    mlx_dtype dtype,
    const mlx_stream s) = NULL;
int (*mlx_flatten_)(
    mlx_array* res,
    const mlx_array a,
    int start_axis,
    int end_axis,
    const mlx_stream s) = NULL;
int (*mlx_floor_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_floor_divide_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_from_fp8_)(
    mlx_array* res,
    const mlx_array x,
    mlx_dtype dtype,
    const mlx_stream s) = NULL;
int (*mlx_full_)(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    const mlx_array vals,
    mlx_dtype dtype,
    const mlx_stream s) = NULL;
int (*mlx_full_like_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array vals,
    mlx_dtype dtype,
    const mlx_stream s) = NULL;
int (*mlx_gather_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_vector_array indices,
    const int* axes,
    size_t axes_num,
    const int* slice_sizes,
    size_t slice_sizes_num,
    const mlx_stream s) = NULL;
int (*mlx_gather_mm_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_array lhs_indices /* may be null */,
    const mlx_array rhs_indices /* may be null */,
    bool sorted_indices,
    const mlx_stream s) = NULL;
int (*mlx_gather_qmm_)(
    mlx_array* res,
    const mlx_array x,
    const mlx_array w,
    const mlx_array scales,
    const mlx_array biases /* may be null */,
    const mlx_array lhs_indices /* may be null */,
    const mlx_array rhs_indices /* may be null */,
    bool transpose,
    mlx_optional_int group_size,
    mlx_optional_int bits,
    const char* mode,
    bool sorted_indices,
    const mlx_stream s) = NULL;
int (*mlx_greater_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_greater_equal_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_hadamard_transform_)(
    mlx_array* res,
    const mlx_array a,
    mlx_optional_float scale,
    const mlx_stream s) = NULL;
int (*mlx_identity_)(mlx_array* res, int n, mlx_dtype dtype, const mlx_stream s) = NULL;
int (*mlx_imag_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_inner_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_isclose_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    double rtol,
    double atol,
    bool equal_nan,
    const mlx_stream s) = NULL;
int (*mlx_isfinite_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_isinf_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_isnan_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_isneginf_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_isposinf_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_kron_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_left_shift_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_less_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_less_equal_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_linspace_)(
    mlx_array* res,
    double start,
    double stop,
    int num,
    mlx_dtype dtype,
    const mlx_stream s) = NULL;
int (*mlx_log_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_log10_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_log1p_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_log2_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_logaddexp_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_logcumsumexp_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool reverse,
    bool inclusive,
    const mlx_stream s) = NULL;
int (*mlx_logical_and_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_logical_not_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_logical_or_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_logsumexp_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_logsumexp_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_logsumexp_)(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_masked_scatter_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array mask,
    const mlx_array src,
    const mlx_stream s) = NULL;
int (*mlx_matmul_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_max_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_max_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_max_)(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_maximum_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_mean_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_mean_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_mean_)(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_median_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_meshgrid_)(
    mlx_vector_array* res,
    const mlx_vector_array arrays,
    bool sparse,
    const char* indexing,
    const mlx_stream s) = NULL;
int (*mlx_min_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_min_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_min_)(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_minimum_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_moveaxis_)(
    mlx_array* res,
    const mlx_array a,
    int source,
    int destination,
    const mlx_stream s) = NULL;
int (*mlx_multiply_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_nan_to_num_)(
    mlx_array* res,
    const mlx_array a,
    float nan,
    mlx_optional_float posinf,
    mlx_optional_float neginf,
    const mlx_stream s) = NULL;
int (*mlx_negative_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_not_equal_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_number_of_elements_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool inverted,
    mlx_dtype dtype,
    const mlx_stream s) = NULL;
int (*mlx_ones_)(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_stream s) = NULL;
int (*mlx_ones_like_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_outer_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_pad_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const int* low_pad_size,
    size_t low_pad_size_num,
    const int* high_pad_size,
    size_t high_pad_size_num,
    const mlx_array pad_value,
    const char* mode,
    const mlx_stream s) = NULL;
int (*mlx_pad_symmetric_)(
    mlx_array* res,
    const mlx_array a,
    int pad_width,
    const mlx_array pad_value,
    const char* mode,
    const mlx_stream s) = NULL;
int (*mlx_partition_axis_)(
    mlx_array* res,
    const mlx_array a,
    int kth,
    int axis,
    const mlx_stream s) = NULL;
int (*mlx_partition_)(
    mlx_array* res,
    const mlx_array a,
    int kth,
    const mlx_stream s) = NULL;
int (*mlx_power_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_prod_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_prod_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_prod_)(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_put_along_axis_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array indices,
    const mlx_array values,
    int axis,
    const mlx_stream s) = NULL;
int (*mlx_quantize_)(
    mlx_vector_array* res,
    const mlx_array w,
    mlx_optional_int group_size,
    mlx_optional_int bits,
    const char* mode,
    const mlx_stream s) = NULL;
int (*mlx_quantized_matmul_)(
    mlx_array* res,
    const mlx_array x,
    const mlx_array w,
    const mlx_array scales,
    const mlx_array biases /* may be null */,
    bool transpose,
    mlx_optional_int group_size,
    mlx_optional_int bits,
    const char* mode,
    const mlx_stream s) = NULL;
int (*mlx_radians_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_real_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_reciprocal_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_remainder_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_repeat_axis_)(
    mlx_array* res,
    const mlx_array arr,
    int repeats,
    int axis,
    const mlx_stream s) = NULL;
int (*mlx_repeat_)(
    mlx_array* res,
    const mlx_array arr,
    int repeats,
    const mlx_stream s) = NULL;
int (*mlx_reshape_)(
    mlx_array* res,
    const mlx_array a,
    const int* shape,
    size_t shape_num,
    const mlx_stream s) = NULL;
int (*mlx_right_shift_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_roll_axis_)(
    mlx_array* res,
    const mlx_array a,
    const int* shift,
    size_t shift_num,
    int axis,
    const mlx_stream s) = NULL;
int (*mlx_roll_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* shift,
    size_t shift_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) = NULL;
int (*mlx_roll_)(
    mlx_array* res,
    const mlx_array a,
    const int* shift,
    size_t shift_num,
    const mlx_stream s) = NULL;
int (*mlx_round_)(
    mlx_array* res,
    const mlx_array a,
    int decimals,
    const mlx_stream s) = NULL;
int (*mlx_rsqrt_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_scatter_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_vector_array indices,
    const mlx_array updates,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) = NULL;
int (*mlx_scatter_add_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_vector_array indices,
    const mlx_array updates,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) = NULL;
int (*mlx_scatter_add_axis_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array indices,
    const mlx_array values,
    int axis,
    const mlx_stream s) = NULL;
int (*mlx_scatter_max_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_vector_array indices,
    const mlx_array updates,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) = NULL;
int (*mlx_scatter_min_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_vector_array indices,
    const mlx_array updates,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) = NULL;
int (*mlx_scatter_prod_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_vector_array indices,
    const mlx_array updates,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) = NULL;
int (*mlx_segmented_mm_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_array segments,
    const mlx_stream s) = NULL;
int (*mlx_sigmoid_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_sign_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_sin_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_sinh_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_slice_)(
    mlx_array* res,
    const mlx_array a,
    const int* start,
    size_t start_num,
    const int* stop,
    size_t stop_num,
    const int* strides,
    size_t strides_num,
    const mlx_stream s) = NULL;
int (*mlx_slice_dynamic_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array start,
    const int* axes,
    size_t axes_num,
    const int* slice_size,
    size_t slice_size_num,
    const mlx_stream s) = NULL;
int (*mlx_slice_update_)(
    mlx_array* res,
    const mlx_array src,
    const mlx_array update,
    const int* start,
    size_t start_num,
    const int* stop,
    size_t stop_num,
    const int* strides,
    size_t strides_num,
    const mlx_stream s) = NULL;
int (*mlx_slice_update_dynamic_)(
    mlx_array* res,
    const mlx_array src,
    const mlx_array update,
    const mlx_array start,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) = NULL;
int (*mlx_softmax_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool precise,
    const mlx_stream s) = NULL;
int (*mlx_softmax_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool precise,
    const mlx_stream s) = NULL;
int (*mlx_softmax_)(
    mlx_array* res,
    const mlx_array a,
    bool precise,
    const mlx_stream s) = NULL;
int (*mlx_sort_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    const mlx_stream s) = NULL;
int (*mlx_sort_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_split_)(
    mlx_vector_array* res,
    const mlx_array a,
    int num_splits,
    int axis,
    const mlx_stream s) = NULL;
int (*mlx_split_sections_)(
    mlx_vector_array* res,
    const mlx_array a,
    const int* indices,
    size_t indices_num,
    int axis,
    const mlx_stream s) = NULL;
int (*mlx_sqrt_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_square_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_squeeze_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) = NULL;
int (*mlx_squeeze_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    const mlx_stream s) = NULL;
int (*mlx_squeeze_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_stack_axis_)(
    mlx_array* res,
    const mlx_vector_array arrays,
    int axis,
    const mlx_stream s) = NULL;
int (*mlx_stack_)(
    mlx_array* res,
    const mlx_vector_array arrays,
    const mlx_stream s) = NULL;
int (*mlx_std_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    int ddof,
    const mlx_stream s) = NULL;
int (*mlx_std_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    int ddof,
    const mlx_stream s) = NULL;
int (*mlx_std_)(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    int ddof,
    const mlx_stream s) = NULL;
int (*mlx_stop_gradient_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_subtract_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) = NULL;
int (*mlx_sum_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_sum_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_sum_)(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s) = NULL;
int (*mlx_swapaxes_)(
    mlx_array* res,
    const mlx_array a,
    int axis1,
    int axis2,
    const mlx_stream s) = NULL;
int (*mlx_take_axis_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array indices,
    int axis,
    const mlx_stream s) = NULL;
int (*mlx_take_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array indices,
    const mlx_stream s) = NULL;
int (*mlx_take_along_axis_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array indices,
    int axis,
    const mlx_stream s) = NULL;
int (*mlx_tan_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_tanh_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_tensordot_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const int* axes_a,
    size_t axes_a_num,
    const int* axes_b,
    size_t axes_b_num,
    const mlx_stream s) = NULL;
int (*mlx_tensordot_axis_)(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    int axis,
    const mlx_stream s) = NULL;
int (*mlx_tile_)(
    mlx_array* res,
    const mlx_array arr,
    const int* reps,
    size_t reps_num,
    const mlx_stream s) = NULL;
int (*mlx_to_fp8_)(mlx_array* res, const mlx_array x, const mlx_stream s) = NULL;
int (*mlx_topk_axis_)(
    mlx_array* res,
    const mlx_array a,
    int k,
    int axis,
    const mlx_stream s) = NULL;
int (*mlx_topk_)(mlx_array* res, const mlx_array a, int k, const mlx_stream s) = NULL;
int (*mlx_trace_)(
    mlx_array* res,
    const mlx_array a,
    int offset,
    int axis1,
    int axis2,
    mlx_dtype dtype,
    const mlx_stream s) = NULL;
int (*mlx_transpose_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) = NULL;
int (*mlx_transpose_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_tri_)(
    mlx_array* res,
    int n,
    int m,
    int k,
    mlx_dtype type,
    const mlx_stream s) = NULL;
int (*mlx_tril_)(mlx_array* res, const mlx_array x, int k, const mlx_stream s) = NULL;
int (*mlx_triu_)(mlx_array* res, const mlx_array x, int k, const mlx_stream s) = NULL;
int (*mlx_unflatten_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    const int* shape,
    size_t shape_num,
    const mlx_stream s) = NULL;
int (*mlx_var_axes_)(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    int ddof,
    const mlx_stream s) = NULL;
int (*mlx_var_axis_)(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    int ddof,
    const mlx_stream s) = NULL;
int (*mlx_var_)(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    int ddof,
    const mlx_stream s) = NULL;
int (*mlx_view_)(
    mlx_array* res,
    const mlx_array a,
    mlx_dtype dtype,
    const mlx_stream s) = NULL;
int (*mlx_where_)(
    mlx_array* res,
    const mlx_array condition,
    const mlx_array x,
    const mlx_array y,
    const mlx_stream s) = NULL;
int (*mlx_zeros_)(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_stream s) = NULL;
int (*mlx_zeros_like_)(mlx_array* res, const mlx_array a, const mlx_stream s) = NULL;
int (*mlx_random_bernoulli_)(
    mlx_array* res,
    const mlx_array p,
    const int* shape,
    size_t shape_num,
    const mlx_array key /* may be null */,
    const mlx_stream s) = NULL;
int (*mlx_random_bits_)(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    int width,
    const mlx_array key /* may be null */,
    const mlx_stream s) = NULL;
int (*mlx_random_categorical_shape_)(
    mlx_array* res,
    const mlx_array logits,
    int axis,
    const int* shape,
    size_t shape_num,
    const mlx_array key /* may be null */,
    const mlx_stream s) = NULL;
int (*mlx_random_categorical_num_samples_)(
    mlx_array* res,
    const mlx_array logits_,
    int axis,
    int num_samples,
    const mlx_array key /* may be null */,
    const mlx_stream s) = NULL;
int (*mlx_random_categorical_)(
    mlx_array* res,
    const mlx_array logits,
    int axis,
    const mlx_array key /* may be null */,
    const mlx_stream s) = NULL;
int (*mlx_random_gumbel_)(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s) = NULL;
int (*mlx_random_key_)(mlx_array* res, uint64_t seed) = NULL;
int (*mlx_random_laplace_)(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    float loc,
    float scale,
    const mlx_array key /* may be null */,
    const mlx_stream s) = NULL;
int (*mlx_random_multivariate_normal_)(
    mlx_array* res,
    const mlx_array mean,
    const mlx_array cov,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s) = NULL;
int (*mlx_random_normal_broadcast_)(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array loc /* may be null */,
    const mlx_array scale /* may be null */,
    const mlx_array key /* may be null */,
    const mlx_stream s) = NULL;
int (*mlx_random_normal_)(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    float loc,
    float scale,
    const mlx_array key /* may be null */,
    const mlx_stream s) = NULL;
int (*mlx_random_permutation_)(
    mlx_array* res,
    const mlx_array x,
    int axis,
    const mlx_array key /* may be null */,
    const mlx_stream s) = NULL;
int (*mlx_random_permutation_arange_)(
    mlx_array* res,
    int x,
    const mlx_array key /* may be null */,
    const mlx_stream s) = NULL;
int (*mlx_random_randint_)(
    mlx_array* res,
    const mlx_array low,
    const mlx_array high,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s) = NULL;
int (*mlx_random_seed_)(uint64_t seed) = NULL;
int (*mlx_random_split_num_)(
    mlx_array* res,
    const mlx_array key,
    int num,
    const mlx_stream s) = NULL;
int (*mlx_random_split_)(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array key,
    const mlx_stream s) = NULL;
int (*mlx_random_truncated_normal_)(
    mlx_array* res,
    const mlx_array lower,
    const mlx_array upper,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s) = NULL;
int (*mlx_random_uniform_)(
    mlx_array* res,
    const mlx_array low,
    const mlx_array high,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s) = NULL;
mlx_stream (*mlx_stream_new_)(void) = NULL;
mlx_stream (*mlx_stream_new_device_)(mlx_device dev) = NULL;
int (*mlx_stream_set_)(mlx_stream* stream, const mlx_stream src) = NULL;
int (*mlx_stream_free_)(mlx_stream stream) = NULL;
int (*mlx_stream_tostring_)(mlx_string* str, mlx_stream stream) = NULL;
bool (*mlx_stream_equal_)(mlx_stream lhs, mlx_stream rhs) = NULL;
int (*mlx_stream_get_device_)(mlx_device* dev, mlx_stream stream) = NULL;
int (*mlx_stream_get_index_)(int* index, mlx_stream stream) = NULL;
int (*mlx_synchronize_)(mlx_stream stream) = NULL;
int (*mlx_get_default_stream_)(mlx_stream* stream, mlx_device dev) = NULL;
int (*mlx_set_default_stream_)(mlx_stream stream) = NULL;
mlx_stream (*mlx_default_cpu_stream_new_)(void) = NULL;
mlx_stream (*mlx_default_gpu_stream_new_)(void) = NULL;
mlx_string (*mlx_string_new_)(void) = NULL;
mlx_string (*mlx_string_new_data_)(const char* str) = NULL;
int (*mlx_string_set_)(mlx_string* str, const mlx_string src) = NULL;
const char * (*mlx_string_data_)(mlx_string str) = NULL;
int (*mlx_string_free_)(mlx_string str) = NULL;
int (*mlx_detail_vmap_replace_)(
    mlx_vector_array* res,
    const mlx_vector_array inputs,
    const mlx_vector_array s_inputs,
    const mlx_vector_array s_outputs,
    const int* in_axes,
    size_t in_axes_num,
    const int* out_axes,
    size_t out_axes_num) = NULL;
int (*mlx_detail_vmap_trace_)(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    const mlx_closure fun,
    const mlx_vector_array inputs,
    const int* in_axes,
    size_t in_axes_num) = NULL;
int (*mlx_async_eval_)(const mlx_vector_array outputs) = NULL;
int (*mlx_checkpoint_)(mlx_closure* res, const mlx_closure fun) = NULL;
int (*mlx_custom_function_)(
    mlx_closure* res,
    const mlx_closure fun,
    const mlx_closure_custom fun_vjp /* may be null */,
    const mlx_closure_custom_jvp fun_jvp /* may be null */,
    const mlx_closure_custom_vmap fun_vmap /* may be null */) = NULL;
int (*mlx_custom_vjp_)(
    mlx_closure* res,
    const mlx_closure fun,
    const mlx_closure_custom fun_vjp) = NULL;
int (*mlx_eval_)(const mlx_vector_array outputs) = NULL;
int (*mlx_jvp_)(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    const mlx_closure fun,
    const mlx_vector_array primals,
    const mlx_vector_array tangents) = NULL;
int (*mlx_value_and_grad_)(
    mlx_closure_value_and_grad* res,
    const mlx_closure fun,
    const int* argnums,
    size_t argnums_num) = NULL;
int (*mlx_vjp_)(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    const mlx_closure fun,
    const mlx_vector_array primals,
    const mlx_vector_array cotangents) = NULL;
mlx_vector_array (*mlx_vector_array_new_)(void) = NULL;
int (*mlx_vector_array_set_)(mlx_vector_array* vec, const mlx_vector_array src) = NULL;
int (*mlx_vector_array_free_)(mlx_vector_array vec) = NULL;
mlx_vector_array (*mlx_vector_array_new_data_)(const mlx_array* data, size_t size) = NULL;
mlx_vector_array (*mlx_vector_array_new_value_)(const mlx_array val) = NULL;
int (*mlx_vector_array_set_data_)(
    mlx_vector_array* vec,
    const mlx_array* data,
    size_t size) = NULL;
int (*mlx_vector_array_set_value_)(mlx_vector_array* vec, const mlx_array val) = NULL;
int (*mlx_vector_array_append_data_)(
    mlx_vector_array vec,
    const mlx_array* data,
    size_t size) = NULL;
int (*mlx_vector_array_append_value_)(mlx_vector_array vec, const mlx_array val) = NULL;
size_t (*mlx_vector_array_size_)(mlx_vector_array vec) = NULL;
int (*mlx_vector_array_get_)(
    mlx_array* res,
    const mlx_vector_array vec,
    size_t idx) = NULL;
mlx_vector_vector_array (*mlx_vector_vector_array_new_)(void) = NULL;
int (*mlx_vector_vector_array_set_)(
    mlx_vector_vector_array* vec,
    const mlx_vector_vector_array src) = NULL;
int (*mlx_vector_vector_array_free_)(mlx_vector_vector_array vec) = NULL;
mlx_vector_vector_array (*mlx_vector_vector_array_new_data_)(
    const mlx_vector_array* data,
    size_t size) = NULL;
mlx_vector_vector_array (*mlx_vector_vector_array_new_value_)(
    const mlx_vector_array val) = NULL;
int (*mlx_vector_vector_array_set_data_)(
    mlx_vector_vector_array* vec,
    const mlx_vector_array* data,
    size_t size) = NULL;
int (*mlx_vector_vector_array_set_value_)(
    mlx_vector_vector_array* vec,
    const mlx_vector_array val) = NULL;
int (*mlx_vector_vector_array_append_data_)(
    mlx_vector_vector_array vec,
    const mlx_vector_array* data,
    size_t size) = NULL;
int (*mlx_vector_vector_array_append_value_)(
    mlx_vector_vector_array vec,
    const mlx_vector_array val) = NULL;
size_t (*mlx_vector_vector_array_size_)(mlx_vector_vector_array vec) = NULL;
int (*mlx_vector_vector_array_get_)(
    mlx_vector_array* res,
    const mlx_vector_vector_array vec,
    size_t idx) = NULL;
mlx_vector_int (*mlx_vector_int_new_)(void) = NULL;
int (*mlx_vector_int_set_)(mlx_vector_int* vec, const mlx_vector_int src) = NULL;
int (*mlx_vector_int_free_)(mlx_vector_int vec) = NULL;
mlx_vector_int (*mlx_vector_int_new_data_)(int* data, size_t size) = NULL;
mlx_vector_int (*mlx_vector_int_new_value_)(int val) = NULL;
int (*mlx_vector_int_set_data_)(mlx_vector_int* vec, int* data, size_t size) = NULL;
int (*mlx_vector_int_set_value_)(mlx_vector_int* vec, int val) = NULL;
int (*mlx_vector_int_append_data_)(mlx_vector_int vec, int* data, size_t size) = NULL;
int (*mlx_vector_int_append_value_)(mlx_vector_int vec, int val) = NULL;
size_t (*mlx_vector_int_size_)(mlx_vector_int vec) = NULL;
int (*mlx_vector_int_get_)(int* res, const mlx_vector_int vec, size_t idx) = NULL;
mlx_vector_string (*mlx_vector_string_new_)(void) = NULL;
int (*mlx_vector_string_set_)(mlx_vector_string* vec, const mlx_vector_string src) = NULL;
int (*mlx_vector_string_free_)(mlx_vector_string vec) = NULL;
mlx_vector_string (*mlx_vector_string_new_data_)(const char** data, size_t size) = NULL;
mlx_vector_string (*mlx_vector_string_new_value_)(const char* val) = NULL;
int (*mlx_vector_string_set_data_)(
    mlx_vector_string* vec,
    const char** data,
    size_t size) = NULL;
int (*mlx_vector_string_set_value_)(mlx_vector_string* vec, const char* val) = NULL;
int (*mlx_vector_string_append_data_)(
    mlx_vector_string vec,
    const char** data,
    size_t size) = NULL;
int (*mlx_vector_string_append_value_)(mlx_vector_string vec, const char* val) = NULL;
size_t (*mlx_vector_string_size_)(mlx_vector_string vec) = NULL;
int (*mlx_vector_string_get_)(char** res, const mlx_vector_string vec, size_t idx) = NULL;
int (*mlx_version_)(mlx_string* str_) = NULL;

int mlx_dynamic_load_symbols(mlx_dynamic_handle handle) {
    CHECK_LOAD(handle, mlx_dtype_size);
    CHECK_LOAD(handle, mlx_array_tostring);
    CHECK_LOAD(handle, mlx_array_new);
    CHECK_LOAD(handle, mlx_array_free);
    CHECK_LOAD(handle, mlx_array_new_bool);
    CHECK_LOAD(handle, mlx_array_new_int);
    CHECK_LOAD(handle, mlx_array_new_float32);
    CHECK_LOAD(handle, mlx_array_new_float);
    CHECK_LOAD(handle, mlx_array_new_float64);
    CHECK_LOAD(handle, mlx_array_new_double);
    CHECK_LOAD(handle, mlx_array_new_complex);
    CHECK_LOAD(handle, mlx_array_new_data);
    CHECK_LOAD(handle, mlx_array_set);
    CHECK_LOAD(handle, mlx_array_set_bool);
    CHECK_LOAD(handle, mlx_array_set_int);
    CHECK_LOAD(handle, mlx_array_set_float32);
    CHECK_LOAD(handle, mlx_array_set_float);
    CHECK_LOAD(handle, mlx_array_set_float64);
    CHECK_LOAD(handle, mlx_array_set_double);
    CHECK_LOAD(handle, mlx_array_set_complex);
    CHECK_LOAD(handle, mlx_array_set_data);
    CHECK_LOAD(handle, mlx_array_itemsize);
    CHECK_LOAD(handle, mlx_array_size);
    CHECK_LOAD(handle, mlx_array_nbytes);
    CHECK_LOAD(handle, mlx_array_ndim);
    CHECK_LOAD(handle, mlx_array_shape);
    CHECK_LOAD(handle, mlx_array_strides);
    CHECK_LOAD(handle, mlx_array_dim);
    CHECK_LOAD(handle, mlx_array_dtype);
    CHECK_LOAD(handle, mlx_array_eval);
    CHECK_LOAD(handle, mlx_array_item_bool);
    CHECK_LOAD(handle, mlx_array_item_uint8);
    CHECK_LOAD(handle, mlx_array_item_uint16);
    CHECK_LOAD(handle, mlx_array_item_uint32);
    CHECK_LOAD(handle, mlx_array_item_uint64);
    CHECK_LOAD(handle, mlx_array_item_int8);
    CHECK_LOAD(handle, mlx_array_item_int16);
    CHECK_LOAD(handle, mlx_array_item_int32);
    CHECK_LOAD(handle, mlx_array_item_int64);
    CHECK_LOAD(handle, mlx_array_item_float32);
    CHECK_LOAD(handle, mlx_array_item_float64);
    CHECK_LOAD(handle, mlx_array_item_complex64);
    CHECK_LOAD(handle, mlx_array_item_float16);
    CHECK_LOAD(handle, mlx_array_item_bfloat16);
    CHECK_LOAD(handle, mlx_array_data_bool);
    CHECK_LOAD(handle, mlx_array_data_uint8);
    CHECK_LOAD(handle, mlx_array_data_uint16);
    CHECK_LOAD(handle, mlx_array_data_uint32);
    CHECK_LOAD(handle, mlx_array_data_uint64);
    CHECK_LOAD(handle, mlx_array_data_int8);
    CHECK_LOAD(handle, mlx_array_data_int16);
    CHECK_LOAD(handle, mlx_array_data_int32);
    CHECK_LOAD(handle, mlx_array_data_int64);
    CHECK_LOAD(handle, mlx_array_data_float32);
    CHECK_LOAD(handle, mlx_array_data_float64);
    CHECK_LOAD(handle, mlx_array_data_complex64);
    CHECK_LOAD(handle, mlx_array_data_float16);
    CHECK_LOAD(handle, mlx_array_data_bfloat16);
    CHECK_LOAD(handle, _mlx_array_is_available);
    CHECK_LOAD(handle, _mlx_array_wait);
    CHECK_LOAD(handle, _mlx_array_is_contiguous);
    CHECK_LOAD(handle, _mlx_array_is_row_contiguous);
    CHECK_LOAD(handle, _mlx_array_is_col_contiguous);
    CHECK_LOAD(handle, mlx_closure_new);
    CHECK_LOAD(handle, mlx_closure_free);
    CHECK_LOAD(handle, mlx_closure_new_func);
    CHECK_LOAD(handle, mlx_closure_new_func_payload);
    CHECK_LOAD(handle, mlx_closure_set);
    CHECK_LOAD(handle, mlx_closure_apply);
    CHECK_LOAD(handle, mlx_closure_new_unary);
    CHECK_LOAD(handle, mlx_closure_kwargs_new);
    CHECK_LOAD(handle, mlx_closure_kwargs_free);
    CHECK_LOAD(handle, mlx_closure_kwargs_new_func);
    CHECK_LOAD(handle, mlx_closure_kwargs_new_func_payload);
    CHECK_LOAD(handle, mlx_closure_kwargs_set);
    CHECK_LOAD(handle, mlx_closure_kwargs_apply);
    CHECK_LOAD(handle, mlx_closure_value_and_grad_new);
    CHECK_LOAD(handle, mlx_closure_value_and_grad_free);
    CHECK_LOAD(handle, mlx_closure_value_and_grad_new_func);
    CHECK_LOAD(handle, mlx_closure_value_and_grad_new_func_payload);
    CHECK_LOAD(handle, mlx_closure_value_and_grad_set);
    CHECK_LOAD(handle, mlx_closure_value_and_grad_apply);
    CHECK_LOAD(handle, mlx_closure_custom_new);
    CHECK_LOAD(handle, mlx_closure_custom_free);
    CHECK_LOAD(handle, mlx_closure_custom_new_func);
    CHECK_LOAD(handle, mlx_closure_custom_new_func_payload);
    CHECK_LOAD(handle, mlx_closure_custom_set);
    CHECK_LOAD(handle, mlx_closure_custom_apply);
    CHECK_LOAD(handle, mlx_closure_custom_jvp_new);
    CHECK_LOAD(handle, mlx_closure_custom_jvp_free);
    CHECK_LOAD(handle, mlx_closure_custom_jvp_new_func);
    CHECK_LOAD(handle, mlx_closure_custom_jvp_new_func_payload);
    CHECK_LOAD(handle, mlx_closure_custom_jvp_set);
    CHECK_LOAD(handle, mlx_closure_custom_jvp_apply);
    CHECK_LOAD(handle, mlx_closure_custom_vmap_new);
    CHECK_LOAD(handle, mlx_closure_custom_vmap_free);
    CHECK_LOAD(handle, mlx_closure_custom_vmap_new_func);
    CHECK_LOAD(handle, mlx_closure_custom_vmap_new_func_payload);
    CHECK_LOAD(handle, mlx_closure_custom_vmap_set);
    CHECK_LOAD(handle, mlx_closure_custom_vmap_apply);
    CHECK_LOAD(handle, mlx_compile);
    CHECK_LOAD(handle, mlx_detail_compile);
    CHECK_LOAD(handle, mlx_detail_compile_clear_cache);
    CHECK_LOAD(handle, mlx_detail_compile_erase);
    CHECK_LOAD(handle, mlx_disable_compile);
    CHECK_LOAD(handle, mlx_enable_compile);
    CHECK_LOAD(handle, mlx_set_compile_mode);
    CHECK_LOAD(handle, mlx_device_new);
    CHECK_LOAD(handle, mlx_device_new_type);
    CHECK_LOAD(handle, mlx_device_free);
    CHECK_LOAD(handle, mlx_device_set);
    CHECK_LOAD(handle, mlx_device_tostring);
    CHECK_LOAD(handle, mlx_device_equal);
    CHECK_LOAD(handle, mlx_device_get_index);
    CHECK_LOAD(handle, mlx_device_get_type);
    CHECK_LOAD(handle, mlx_get_default_device);
    CHECK_LOAD(handle, mlx_set_default_device);
    CHECK_LOAD(handle, mlx_distributed_group_rank);
    CHECK_LOAD(handle, mlx_distributed_group_size);
    CHECK_LOAD(handle, mlx_distributed_group_split);
    CHECK_LOAD(handle, mlx_distributed_is_available);
    CHECK_LOAD(handle, mlx_distributed_init);
    CHECK_LOAD(handle, mlx_distributed_all_gather);
    CHECK_LOAD(handle, mlx_distributed_all_max);
    CHECK_LOAD(handle, mlx_distributed_all_min);
    CHECK_LOAD(handle, mlx_distributed_all_sum);
    CHECK_LOAD(handle, mlx_distributed_recv);
    CHECK_LOAD(handle, mlx_distributed_recv_like);
    CHECK_LOAD(handle, mlx_distributed_send);
    CHECK_LOAD(handle, mlx_distributed_sum_scatter);
    CHECK_LOAD(handle, mlx_set_error_handler);
    CHECK_LOAD(handle, _mlx_error);
    CHECK_LOAD(handle, mlx_export_function);
    CHECK_LOAD(handle, mlx_export_function_kwargs);
    CHECK_LOAD(handle, mlx_function_exporter_new);
    CHECK_LOAD(handle, mlx_function_exporter_free);
    CHECK_LOAD(handle, mlx_function_exporter_apply);
    CHECK_LOAD(handle, mlx_function_exporter_apply_kwargs);
    CHECK_LOAD(handle, mlx_imported_function_new);
    CHECK_LOAD(handle, mlx_imported_function_free);
    CHECK_LOAD(handle, mlx_imported_function_apply);
    CHECK_LOAD(handle, mlx_imported_function_apply_kwargs);
    CHECK_LOAD(handle, mlx_fast_cuda_kernel_config_new);
    CHECK_LOAD(handle, mlx_fast_cuda_kernel_config_free);
    CHECK_LOAD(handle, mlx_fast_cuda_kernel_config_add_output_arg);
    CHECK_LOAD(handle, mlx_fast_cuda_kernel_config_set_grid);
    CHECK_LOAD(handle, mlx_fast_cuda_kernel_config_set_thread_group);
    CHECK_LOAD(handle, mlx_fast_cuda_kernel_config_set_init_value);
    CHECK_LOAD(handle, mlx_fast_cuda_kernel_config_set_verbose);
    CHECK_LOAD(handle, mlx_fast_cuda_kernel_config_add_template_arg_dtype);
    CHECK_LOAD(handle, mlx_fast_cuda_kernel_config_add_template_arg_int);
    CHECK_LOAD(handle, mlx_fast_cuda_kernel_config_add_template_arg_bool);
    CHECK_LOAD(handle, mlx_fast_cuda_kernel_new);
    CHECK_LOAD(handle, mlx_fast_cuda_kernel_free);
    CHECK_LOAD(handle, mlx_fast_cuda_kernel_apply);
    CHECK_LOAD(handle, mlx_fast_layer_norm);
    CHECK_LOAD(handle, mlx_fast_metal_kernel_config_new);
    CHECK_LOAD(handle, mlx_fast_metal_kernel_config_free);
    CHECK_LOAD(handle, mlx_fast_metal_kernel_config_add_output_arg);
    CHECK_LOAD(handle, mlx_fast_metal_kernel_config_set_grid);
    CHECK_LOAD(handle, mlx_fast_metal_kernel_config_set_thread_group);
    CHECK_LOAD(handle, mlx_fast_metal_kernel_config_set_init_value);
    CHECK_LOAD(handle, mlx_fast_metal_kernel_config_set_verbose);
    CHECK_LOAD(handle, mlx_fast_metal_kernel_config_add_template_arg_dtype);
    CHECK_LOAD(handle, mlx_fast_metal_kernel_config_add_template_arg_int);
    CHECK_LOAD(handle, mlx_fast_metal_kernel_config_add_template_arg_bool);
    CHECK_LOAD(handle, mlx_fast_metal_kernel_new);
    CHECK_LOAD(handle, mlx_fast_metal_kernel_free);
    CHECK_LOAD(handle, mlx_fast_metal_kernel_apply);
    CHECK_LOAD(handle, mlx_fast_rms_norm);
    CHECK_LOAD(handle, mlx_fast_rope);
    CHECK_LOAD(handle, mlx_fast_scaled_dot_product_attention);
    CHECK_LOAD(handle, mlx_fft_fft);
    CHECK_LOAD(handle, mlx_fft_fft2);
    CHECK_LOAD(handle, mlx_fft_fftn);
    CHECK_LOAD(handle, mlx_fft_fftshift);
    CHECK_LOAD(handle, mlx_fft_ifft);
    CHECK_LOAD(handle, mlx_fft_ifft2);
    CHECK_LOAD(handle, mlx_fft_ifftn);
    CHECK_LOAD(handle, mlx_fft_ifftshift);
    CHECK_LOAD(handle, mlx_fft_irfft);
    CHECK_LOAD(handle, mlx_fft_irfft2);
    CHECK_LOAD(handle, mlx_fft_irfftn);
    CHECK_LOAD(handle, mlx_fft_rfft);
    CHECK_LOAD(handle, mlx_fft_rfft2);
    CHECK_LOAD(handle, mlx_fft_rfftn);
    CHECK_LOAD(handle, mlx_io_reader_new);
    CHECK_LOAD(handle, mlx_io_reader_descriptor);
    CHECK_LOAD(handle, mlx_io_reader_tostring);
    CHECK_LOAD(handle, mlx_io_reader_free);
    CHECK_LOAD(handle, mlx_io_writer_new);
    CHECK_LOAD(handle, mlx_io_writer_descriptor);
    CHECK_LOAD(handle, mlx_io_writer_tostring);
    CHECK_LOAD(handle, mlx_io_writer_free);
    CHECK_LOAD(handle, mlx_load_reader);
    CHECK_LOAD(handle, mlx_load);
    CHECK_LOAD(handle, mlx_load_safetensors_reader);
    CHECK_LOAD(handle, mlx_load_safetensors);
    CHECK_LOAD(handle, mlx_save_writer);
    CHECK_LOAD(handle, mlx_save);
    CHECK_LOAD(handle, mlx_save_safetensors_writer);
    CHECK_LOAD(handle, mlx_save_safetensors);
    CHECK_LOAD(handle, mlx_linalg_cholesky);
    CHECK_LOAD(handle, mlx_linalg_cholesky_inv);
    CHECK_LOAD(handle, mlx_linalg_cross);
    CHECK_LOAD(handle, mlx_linalg_eig);
    CHECK_LOAD(handle, mlx_linalg_eigh);
    CHECK_LOAD(handle, mlx_linalg_eigvals);
    CHECK_LOAD(handle, mlx_linalg_eigvalsh);
    CHECK_LOAD(handle, mlx_linalg_inv);
    CHECK_LOAD(handle, mlx_linalg_lu);
    CHECK_LOAD(handle, mlx_linalg_lu_factor);
    CHECK_LOAD(handle, mlx_linalg_norm);
    CHECK_LOAD(handle, mlx_linalg_norm_matrix);
    CHECK_LOAD(handle, mlx_linalg_norm_l2);
    CHECK_LOAD(handle, mlx_linalg_pinv);
    CHECK_LOAD(handle, mlx_linalg_qr);
    CHECK_LOAD(handle, mlx_linalg_solve);
    CHECK_LOAD(handle, mlx_linalg_solve_triangular);
    CHECK_LOAD(handle, mlx_linalg_svd);
    CHECK_LOAD(handle, mlx_linalg_tri_inv);
    CHECK_LOAD(handle, mlx_map_string_to_array_new);
    CHECK_LOAD(handle, mlx_map_string_to_array_set);
    CHECK_LOAD(handle, mlx_map_string_to_array_free);
    CHECK_LOAD(handle, mlx_map_string_to_array_insert);
    CHECK_LOAD(handle, mlx_map_string_to_array_get);
    CHECK_LOAD(handle, mlx_map_string_to_array_iterator_new);
    CHECK_LOAD(handle, mlx_map_string_to_array_iterator_free);
    CHECK_LOAD(handle, mlx_map_string_to_array_iterator_next);
    CHECK_LOAD(handle, mlx_map_string_to_string_new);
    CHECK_LOAD(handle, mlx_map_string_to_string_set);
    CHECK_LOAD(handle, mlx_map_string_to_string_free);
    CHECK_LOAD(handle, mlx_map_string_to_string_insert);
    CHECK_LOAD(handle, mlx_map_string_to_string_get);
    CHECK_LOAD(handle, mlx_map_string_to_string_iterator_new);
    CHECK_LOAD(handle, mlx_map_string_to_string_iterator_free);
    CHECK_LOAD(handle, mlx_map_string_to_string_iterator_next);
    CHECK_LOAD(handle, mlx_clear_cache);
    CHECK_LOAD(handle, mlx_get_active_memory);
    CHECK_LOAD(handle, mlx_get_cache_memory);
    CHECK_LOAD(handle, mlx_get_memory_limit);
    CHECK_LOAD(handle, mlx_get_peak_memory);
    CHECK_LOAD(handle, mlx_reset_peak_memory);
    CHECK_LOAD(handle, mlx_set_cache_limit);
    CHECK_LOAD(handle, mlx_set_memory_limit);
    CHECK_LOAD(handle, mlx_set_wired_limit);
    CHECK_LOAD(handle, mlx_metal_device_info);
    CHECK_LOAD(handle, mlx_metal_is_available);
    CHECK_LOAD(handle, mlx_metal_start_capture);
    CHECK_LOAD(handle, mlx_metal_stop_capture);
    CHECK_LOAD(handle, mlx_abs);
    CHECK_LOAD(handle, mlx_add);
    CHECK_LOAD(handle, mlx_addmm);
    CHECK_LOAD(handle, mlx_all_axes);
    CHECK_LOAD(handle, mlx_all_axis);
    CHECK_LOAD(handle, mlx_all);
    CHECK_LOAD(handle, mlx_allclose);
    CHECK_LOAD(handle, mlx_any_axes);
    CHECK_LOAD(handle, mlx_any_axis);
    CHECK_LOAD(handle, mlx_any);
    CHECK_LOAD(handle, mlx_arange);
    CHECK_LOAD(handle, mlx_arccos);
    CHECK_LOAD(handle, mlx_arccosh);
    CHECK_LOAD(handle, mlx_arcsin);
    CHECK_LOAD(handle, mlx_arcsinh);
    CHECK_LOAD(handle, mlx_arctan);
    CHECK_LOAD(handle, mlx_arctan2);
    CHECK_LOAD(handle, mlx_arctanh);
    CHECK_LOAD(handle, mlx_argmax_axis);
    CHECK_LOAD(handle, mlx_argmax);
    CHECK_LOAD(handle, mlx_argmin_axis);
    CHECK_LOAD(handle, mlx_argmin);
    CHECK_LOAD(handle, mlx_argpartition_axis);
    CHECK_LOAD(handle, mlx_argpartition);
    CHECK_LOAD(handle, mlx_argsort_axis);
    CHECK_LOAD(handle, mlx_argsort);
    CHECK_LOAD(handle, mlx_array_equal);
    CHECK_LOAD(handle, mlx_as_strided);
    CHECK_LOAD(handle, mlx_astype);
    CHECK_LOAD(handle, mlx_atleast_1d);
    CHECK_LOAD(handle, mlx_atleast_2d);
    CHECK_LOAD(handle, mlx_atleast_3d);
    CHECK_LOAD(handle, mlx_bitwise_and);
    CHECK_LOAD(handle, mlx_bitwise_invert);
    CHECK_LOAD(handle, mlx_bitwise_or);
    CHECK_LOAD(handle, mlx_bitwise_xor);
    CHECK_LOAD(handle, mlx_block_masked_mm);
    CHECK_LOAD(handle, mlx_broadcast_arrays);
    CHECK_LOAD(handle, mlx_broadcast_to);
    CHECK_LOAD(handle, mlx_ceil);
    CHECK_LOAD(handle, mlx_clip);
    CHECK_LOAD(handle, mlx_concatenate_axis);
    CHECK_LOAD(handle, mlx_concatenate);
    CHECK_LOAD(handle, mlx_conjugate);
    CHECK_LOAD(handle, mlx_contiguous);
    CHECK_LOAD(handle, mlx_conv1d);
    CHECK_LOAD(handle, mlx_conv2d);
    CHECK_LOAD(handle, mlx_conv3d);
    CHECK_LOAD(handle, mlx_conv_general);
    CHECK_LOAD(handle, mlx_conv_transpose1d);
    CHECK_LOAD(handle, mlx_conv_transpose2d);
    CHECK_LOAD(handle, mlx_conv_transpose3d);
    CHECK_LOAD(handle, mlx_copy);
    CHECK_LOAD(handle, mlx_cos);
    CHECK_LOAD(handle, mlx_cosh);
    CHECK_LOAD(handle, mlx_cummax);
    CHECK_LOAD(handle, mlx_cummin);
    CHECK_LOAD(handle, mlx_cumprod);
    CHECK_LOAD(handle, mlx_cumsum);
    CHECK_LOAD(handle, mlx_degrees);
    CHECK_LOAD(handle, mlx_depends);
    CHECK_LOAD(handle, mlx_dequantize);
    CHECK_LOAD(handle, mlx_diag);
    CHECK_LOAD(handle, mlx_diagonal);
    CHECK_LOAD(handle, mlx_divide);
    CHECK_LOAD(handle, mlx_divmod);
    CHECK_LOAD(handle, mlx_einsum);
    CHECK_LOAD(handle, mlx_equal);
    CHECK_LOAD(handle, mlx_erf);
    CHECK_LOAD(handle, mlx_erfinv);
    CHECK_LOAD(handle, mlx_exp);
    CHECK_LOAD(handle, mlx_expand_dims_axes);
    CHECK_LOAD(handle, mlx_expand_dims);
    CHECK_LOAD(handle, mlx_expm1);
    CHECK_LOAD(handle, mlx_eye);
    CHECK_LOAD(handle, mlx_flatten);
    CHECK_LOAD(handle, mlx_floor);
    CHECK_LOAD(handle, mlx_floor_divide);
    CHECK_LOAD(handle, mlx_from_fp8);
    CHECK_LOAD(handle, mlx_full);
    CHECK_LOAD(handle, mlx_full_like);
    CHECK_LOAD(handle, mlx_gather);
    CHECK_LOAD(handle, mlx_gather_mm);
    CHECK_LOAD(handle, mlx_gather_qmm);
    CHECK_LOAD(handle, mlx_greater);
    CHECK_LOAD(handle, mlx_greater_equal);
    CHECK_LOAD(handle, mlx_hadamard_transform);
    CHECK_LOAD(handle, mlx_identity);
    CHECK_LOAD(handle, mlx_imag);
    CHECK_LOAD(handle, mlx_inner);
    CHECK_LOAD(handle, mlx_isclose);
    CHECK_LOAD(handle, mlx_isfinite);
    CHECK_LOAD(handle, mlx_isinf);
    CHECK_LOAD(handle, mlx_isnan);
    CHECK_LOAD(handle, mlx_isneginf);
    CHECK_LOAD(handle, mlx_isposinf);
    CHECK_LOAD(handle, mlx_kron);
    CHECK_LOAD(handle, mlx_left_shift);
    CHECK_LOAD(handle, mlx_less);
    CHECK_LOAD(handle, mlx_less_equal);
    CHECK_LOAD(handle, mlx_linspace);
    CHECK_LOAD(handle, mlx_log);
    CHECK_LOAD(handle, mlx_log10);
    CHECK_LOAD(handle, mlx_log1p);
    CHECK_LOAD(handle, mlx_log2);
    CHECK_LOAD(handle, mlx_logaddexp);
    CHECK_LOAD(handle, mlx_logcumsumexp);
    CHECK_LOAD(handle, mlx_logical_and);
    CHECK_LOAD(handle, mlx_logical_not);
    CHECK_LOAD(handle, mlx_logical_or);
    CHECK_LOAD(handle, mlx_logsumexp_axes);
    CHECK_LOAD(handle, mlx_logsumexp_axis);
    CHECK_LOAD(handle, mlx_logsumexp);
    CHECK_LOAD(handle, mlx_masked_scatter);
    CHECK_LOAD(handle, mlx_matmul);
    CHECK_LOAD(handle, mlx_max_axes);
    CHECK_LOAD(handle, mlx_max_axis);
    CHECK_LOAD(handle, mlx_max);
    CHECK_LOAD(handle, mlx_maximum);
    CHECK_LOAD(handle, mlx_mean_axes);
    CHECK_LOAD(handle, mlx_mean_axis);
    CHECK_LOAD(handle, mlx_mean);
    CHECK_LOAD(handle, mlx_median);
    CHECK_LOAD(handle, mlx_meshgrid);
    CHECK_LOAD(handle, mlx_min_axes);
    CHECK_LOAD(handle, mlx_min_axis);
    CHECK_LOAD(handle, mlx_min);
    CHECK_LOAD(handle, mlx_minimum);
    CHECK_LOAD(handle, mlx_moveaxis);
    CHECK_LOAD(handle, mlx_multiply);
    CHECK_LOAD(handle, mlx_nan_to_num);
    CHECK_LOAD(handle, mlx_negative);
    CHECK_LOAD(handle, mlx_not_equal);
    CHECK_LOAD(handle, mlx_number_of_elements);
    CHECK_LOAD(handle, mlx_ones);
    CHECK_LOAD(handle, mlx_ones_like);
    CHECK_LOAD(handle, mlx_outer);
    CHECK_LOAD(handle, mlx_pad);
    CHECK_LOAD(handle, mlx_pad_symmetric);
    CHECK_LOAD(handle, mlx_partition_axis);
    CHECK_LOAD(handle, mlx_partition);
    CHECK_LOAD(handle, mlx_power);
    CHECK_LOAD(handle, mlx_prod_axes);
    CHECK_LOAD(handle, mlx_prod_axis);
    CHECK_LOAD(handle, mlx_prod);
    CHECK_LOAD(handle, mlx_put_along_axis);
    CHECK_LOAD(handle, mlx_quantize);
    CHECK_LOAD(handle, mlx_quantized_matmul);
    CHECK_LOAD(handle, mlx_radians);
    CHECK_LOAD(handle, mlx_real);
    CHECK_LOAD(handle, mlx_reciprocal);
    CHECK_LOAD(handle, mlx_remainder);
    CHECK_LOAD(handle, mlx_repeat_axis);
    CHECK_LOAD(handle, mlx_repeat);
    CHECK_LOAD(handle, mlx_reshape);
    CHECK_LOAD(handle, mlx_right_shift);
    CHECK_LOAD(handle, mlx_roll_axis);
    CHECK_LOAD(handle, mlx_roll_axes);
    CHECK_LOAD(handle, mlx_roll);
    CHECK_LOAD(handle, mlx_round);
    CHECK_LOAD(handle, mlx_rsqrt);
    CHECK_LOAD(handle, mlx_scatter);
    CHECK_LOAD(handle, mlx_scatter_add);
    CHECK_LOAD(handle, mlx_scatter_add_axis);
    CHECK_LOAD(handle, mlx_scatter_max);
    CHECK_LOAD(handle, mlx_scatter_min);
    CHECK_LOAD(handle, mlx_scatter_prod);
    CHECK_LOAD(handle, mlx_segmented_mm);
    CHECK_LOAD(handle, mlx_sigmoid);
    CHECK_LOAD(handle, mlx_sign);
    CHECK_LOAD(handle, mlx_sin);
    CHECK_LOAD(handle, mlx_sinh);
    CHECK_LOAD(handle, mlx_slice);
    CHECK_LOAD(handle, mlx_slice_dynamic);
    CHECK_LOAD(handle, mlx_slice_update);
    CHECK_LOAD(handle, mlx_slice_update_dynamic);
    CHECK_LOAD(handle, mlx_softmax_axes);
    CHECK_LOAD(handle, mlx_softmax_axis);
    CHECK_LOAD(handle, mlx_softmax);
    CHECK_LOAD(handle, mlx_sort_axis);
    CHECK_LOAD(handle, mlx_sort);
    CHECK_LOAD(handle, mlx_split);
    CHECK_LOAD(handle, mlx_split_sections);
    CHECK_LOAD(handle, mlx_sqrt);
    CHECK_LOAD(handle, mlx_square);
    CHECK_LOAD(handle, mlx_squeeze_axes);
    CHECK_LOAD(handle, mlx_squeeze_axis);
    CHECK_LOAD(handle, mlx_squeeze);
    CHECK_LOAD(handle, mlx_stack_axis);
    CHECK_LOAD(handle, mlx_stack);
    CHECK_LOAD(handle, mlx_std_axes);
    CHECK_LOAD(handle, mlx_std_axis);
    CHECK_LOAD(handle, mlx_std);
    CHECK_LOAD(handle, mlx_stop_gradient);
    CHECK_LOAD(handle, mlx_subtract);
    CHECK_LOAD(handle, mlx_sum_axes);
    CHECK_LOAD(handle, mlx_sum_axis);
    CHECK_LOAD(handle, mlx_sum);
    CHECK_LOAD(handle, mlx_swapaxes);
    CHECK_LOAD(handle, mlx_take_axis);
    CHECK_LOAD(handle, mlx_take);
    CHECK_LOAD(handle, mlx_take_along_axis);
    CHECK_LOAD(handle, mlx_tan);
    CHECK_LOAD(handle, mlx_tanh);
    CHECK_LOAD(handle, mlx_tensordot);
    CHECK_LOAD(handle, mlx_tensordot_axis);
    CHECK_LOAD(handle, mlx_tile);
    CHECK_LOAD(handle, mlx_to_fp8);
    CHECK_LOAD(handle, mlx_topk_axis);
    CHECK_LOAD(handle, mlx_topk);
    CHECK_LOAD(handle, mlx_trace);
    CHECK_LOAD(handle, mlx_transpose_axes);
    CHECK_LOAD(handle, mlx_transpose);
    CHECK_LOAD(handle, mlx_tri);
    CHECK_LOAD(handle, mlx_tril);
    CHECK_LOAD(handle, mlx_triu);
    CHECK_LOAD(handle, mlx_unflatten);
    CHECK_LOAD(handle, mlx_var_axes);
    CHECK_LOAD(handle, mlx_var_axis);
    CHECK_LOAD(handle, mlx_var);
    CHECK_LOAD(handle, mlx_view);
    CHECK_LOAD(handle, mlx_where);
    CHECK_LOAD(handle, mlx_zeros);
    CHECK_LOAD(handle, mlx_zeros_like);
    CHECK_LOAD(handle, mlx_random_bernoulli);
    CHECK_LOAD(handle, mlx_random_bits);
    CHECK_LOAD(handle, mlx_random_categorical_shape);
    CHECK_LOAD(handle, mlx_random_categorical_num_samples);
    CHECK_LOAD(handle, mlx_random_categorical);
    CHECK_LOAD(handle, mlx_random_gumbel);
    CHECK_LOAD(handle, mlx_random_key);
    CHECK_LOAD(handle, mlx_random_laplace);
    CHECK_LOAD(handle, mlx_random_multivariate_normal);
    CHECK_LOAD(handle, mlx_random_normal_broadcast);
    CHECK_LOAD(handle, mlx_random_normal);
    CHECK_LOAD(handle, mlx_random_permutation);
    CHECK_LOAD(handle, mlx_random_permutation_arange);
    CHECK_LOAD(handle, mlx_random_randint);
    CHECK_LOAD(handle, mlx_random_seed);
    CHECK_LOAD(handle, mlx_random_split_num);
    CHECK_LOAD(handle, mlx_random_split);
    CHECK_LOAD(handle, mlx_random_truncated_normal);
    CHECK_LOAD(handle, mlx_random_uniform);
    CHECK_LOAD(handle, mlx_stream_new);
    CHECK_LOAD(handle, mlx_stream_new_device);
    CHECK_LOAD(handle, mlx_stream_set);
    CHECK_LOAD(handle, mlx_stream_free);
    CHECK_LOAD(handle, mlx_stream_tostring);
    CHECK_LOAD(handle, mlx_stream_equal);
    CHECK_LOAD(handle, mlx_stream_get_device);
    CHECK_LOAD(handle, mlx_stream_get_index);
    CHECK_LOAD(handle, mlx_synchronize);
    CHECK_LOAD(handle, mlx_get_default_stream);
    CHECK_LOAD(handle, mlx_set_default_stream);
    CHECK_LOAD(handle, mlx_default_cpu_stream_new);
    CHECK_LOAD(handle, mlx_default_gpu_stream_new);
    CHECK_LOAD(handle, mlx_string_new);
    CHECK_LOAD(handle, mlx_string_new_data);
    CHECK_LOAD(handle, mlx_string_set);
    CHECK_LOAD(handle, mlx_string_data);
    CHECK_LOAD(handle, mlx_string_free);
    CHECK_LOAD(handle, mlx_detail_vmap_replace);
    CHECK_LOAD(handle, mlx_detail_vmap_trace);
    CHECK_LOAD(handle, mlx_async_eval);
    CHECK_LOAD(handle, mlx_checkpoint);
    CHECK_LOAD(handle, mlx_custom_function);
    CHECK_LOAD(handle, mlx_custom_vjp);
    CHECK_LOAD(handle, mlx_eval);
    CHECK_LOAD(handle, mlx_jvp);
    CHECK_LOAD(handle, mlx_value_and_grad);
    CHECK_LOAD(handle, mlx_vjp);
    CHECK_LOAD(handle, mlx_vector_array_new);
    CHECK_LOAD(handle, mlx_vector_array_set);
    CHECK_LOAD(handle, mlx_vector_array_free);
    CHECK_LOAD(handle, mlx_vector_array_new_data);
    CHECK_LOAD(handle, mlx_vector_array_new_value);
    CHECK_LOAD(handle, mlx_vector_array_set_data);
    CHECK_LOAD(handle, mlx_vector_array_set_value);
    CHECK_LOAD(handle, mlx_vector_array_append_data);
    CHECK_LOAD(handle, mlx_vector_array_append_value);
    CHECK_LOAD(handle, mlx_vector_array_size);
    CHECK_LOAD(handle, mlx_vector_array_get);
    CHECK_LOAD(handle, mlx_vector_vector_array_new);
    CHECK_LOAD(handle, mlx_vector_vector_array_set);
    CHECK_LOAD(handle, mlx_vector_vector_array_free);
    CHECK_LOAD(handle, mlx_vector_vector_array_new_data);
    CHECK_LOAD(handle, mlx_vector_vector_array_new_value);
    CHECK_LOAD(handle, mlx_vector_vector_array_set_data);
    CHECK_LOAD(handle, mlx_vector_vector_array_set_value);
    CHECK_LOAD(handle, mlx_vector_vector_array_append_data);
    CHECK_LOAD(handle, mlx_vector_vector_array_append_value);
    CHECK_LOAD(handle, mlx_vector_vector_array_size);
    CHECK_LOAD(handle, mlx_vector_vector_array_get);
    CHECK_LOAD(handle, mlx_vector_int_new);
    CHECK_LOAD(handle, mlx_vector_int_set);
    CHECK_LOAD(handle, mlx_vector_int_free);
    CHECK_LOAD(handle, mlx_vector_int_new_data);
    CHECK_LOAD(handle, mlx_vector_int_new_value);
    CHECK_LOAD(handle, mlx_vector_int_set_data);
    CHECK_LOAD(handle, mlx_vector_int_set_value);
    CHECK_LOAD(handle, mlx_vector_int_append_data);
    CHECK_LOAD(handle, mlx_vector_int_append_value);
    CHECK_LOAD(handle, mlx_vector_int_size);
    CHECK_LOAD(handle, mlx_vector_int_get);
    CHECK_LOAD(handle, mlx_vector_string_new);
    CHECK_LOAD(handle, mlx_vector_string_set);
    CHECK_LOAD(handle, mlx_vector_string_free);
    CHECK_LOAD(handle, mlx_vector_string_new_data);
    CHECK_LOAD(handle, mlx_vector_string_new_value);
    CHECK_LOAD(handle, mlx_vector_string_set_data);
    CHECK_LOAD(handle, mlx_vector_string_set_value);
    CHECK_LOAD(handle, mlx_vector_string_append_data);
    CHECK_LOAD(handle, mlx_vector_string_append_value);
    CHECK_LOAD(handle, mlx_vector_string_size);
    CHECK_LOAD(handle, mlx_vector_string_get);
    CHECK_LOAD(handle, mlx_version);
    return 0;
}
