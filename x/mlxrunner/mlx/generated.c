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

size_t mlx_dtype_size(mlx_dtype dtype) {
    return mlx_dtype_size_(dtype);
}

int mlx_array_tostring(mlx_string* str, const mlx_array arr) {
    return mlx_array_tostring_(str, arr);
}

mlx_array mlx_array_new(void) {
    return mlx_array_new_();
}

int mlx_array_free(mlx_array arr) {
    return mlx_array_free_(arr);
}

mlx_array mlx_array_new_bool(bool val) {
    return mlx_array_new_bool_(val);
}

mlx_array mlx_array_new_int(int val) {
    return mlx_array_new_int_(val);
}

mlx_array mlx_array_new_float32(float val) {
    return mlx_array_new_float32_(val);
}

mlx_array mlx_array_new_float(float val) {
    return mlx_array_new_float_(val);
}

mlx_array mlx_array_new_float64(double val) {
    return mlx_array_new_float64_(val);
}

mlx_array mlx_array_new_double(double val) {
    return mlx_array_new_double_(val);
}

mlx_array mlx_array_new_complex(float real_val, float imag_val) {
    return mlx_array_new_complex_(real_val, imag_val);
}

mlx_array mlx_array_new_data(
    const void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype) {
    return mlx_array_new_data_(data, shape, dim, dtype);
}

int mlx_array_set(mlx_array* arr, const mlx_array src) {
    return mlx_array_set_(arr, src);
}

int mlx_array_set_bool(mlx_array* arr, bool val) {
    return mlx_array_set_bool_(arr, val);
}

int mlx_array_set_int(mlx_array* arr, int val) {
    return mlx_array_set_int_(arr, val);
}

int mlx_array_set_float32(mlx_array* arr, float val) {
    return mlx_array_set_float32_(arr, val);
}

int mlx_array_set_float(mlx_array* arr, float val) {
    return mlx_array_set_float_(arr, val);
}

int mlx_array_set_float64(mlx_array* arr, double val) {
    return mlx_array_set_float64_(arr, val);
}

int mlx_array_set_double(mlx_array* arr, double val) {
    return mlx_array_set_double_(arr, val);
}

int mlx_array_set_complex(mlx_array* arr, float real_val, float imag_val) {
    return mlx_array_set_complex_(arr, real_val, imag_val);
}

int mlx_array_set_data(
    mlx_array* arr,
    const void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype) {
    return mlx_array_set_data_(arr, data, shape, dim, dtype);
}

size_t mlx_array_itemsize(const mlx_array arr) {
    return mlx_array_itemsize_(arr);
}

size_t mlx_array_size(const mlx_array arr) {
    return mlx_array_size_(arr);
}

size_t mlx_array_nbytes(const mlx_array arr) {
    return mlx_array_nbytes_(arr);
}

size_t mlx_array_ndim(const mlx_array arr) {
    return mlx_array_ndim_(arr);
}

const int * mlx_array_shape(const mlx_array arr) {
    return mlx_array_shape_(arr);
}

const size_t * mlx_array_strides(const mlx_array arr) {
    return mlx_array_strides_(arr);
}

int mlx_array_dim(const mlx_array arr, int dim) {
    return mlx_array_dim_(arr, dim);
}

mlx_dtype mlx_array_dtype(const mlx_array arr) {
    return mlx_array_dtype_(arr);
}

int mlx_array_eval(mlx_array arr) {
    return mlx_array_eval_(arr);
}

int mlx_array_item_bool(bool* res, const mlx_array arr) {
    return mlx_array_item_bool_(res, arr);
}

int mlx_array_item_uint8(uint8_t* res, const mlx_array arr) {
    return mlx_array_item_uint8_(res, arr);
}

int mlx_array_item_uint16(uint16_t* res, const mlx_array arr) {
    return mlx_array_item_uint16_(res, arr);
}

int mlx_array_item_uint32(uint32_t* res, const mlx_array arr) {
    return mlx_array_item_uint32_(res, arr);
}

int mlx_array_item_uint64(uint64_t* res, const mlx_array arr) {
    return mlx_array_item_uint64_(res, arr);
}

int mlx_array_item_int8(int8_t* res, const mlx_array arr) {
    return mlx_array_item_int8_(res, arr);
}

int mlx_array_item_int16(int16_t* res, const mlx_array arr) {
    return mlx_array_item_int16_(res, arr);
}

int mlx_array_item_int32(int32_t* res, const mlx_array arr) {
    return mlx_array_item_int32_(res, arr);
}

int mlx_array_item_int64(int64_t* res, const mlx_array arr) {
    return mlx_array_item_int64_(res, arr);
}

int mlx_array_item_float32(float* res, const mlx_array arr) {
    return mlx_array_item_float32_(res, arr);
}

int mlx_array_item_float64(double* res, const mlx_array arr) {
    return mlx_array_item_float64_(res, arr);
}

int mlx_array_item_complex64(float _Complex* res, const mlx_array arr) {
    return mlx_array_item_complex64_(res, arr);
}

int mlx_array_item_float16(float16_t* res, const mlx_array arr) {
    return mlx_array_item_float16_(res, arr);
}

int mlx_array_item_bfloat16(bfloat16_t* res, const mlx_array arr) {
    return mlx_array_item_bfloat16_(res, arr);
}

const bool * mlx_array_data_bool(const mlx_array arr) {
    return mlx_array_data_bool_(arr);
}

const uint8_t * mlx_array_data_uint8(const mlx_array arr) {
    return mlx_array_data_uint8_(arr);
}

const uint16_t * mlx_array_data_uint16(const mlx_array arr) {
    return mlx_array_data_uint16_(arr);
}

const uint32_t * mlx_array_data_uint32(const mlx_array arr) {
    return mlx_array_data_uint32_(arr);
}

const uint64_t * mlx_array_data_uint64(const mlx_array arr) {
    return mlx_array_data_uint64_(arr);
}

const int8_t * mlx_array_data_int8(const mlx_array arr) {
    return mlx_array_data_int8_(arr);
}

const int16_t * mlx_array_data_int16(const mlx_array arr) {
    return mlx_array_data_int16_(arr);
}

const int32_t * mlx_array_data_int32(const mlx_array arr) {
    return mlx_array_data_int32_(arr);
}

const int64_t * mlx_array_data_int64(const mlx_array arr) {
    return mlx_array_data_int64_(arr);
}

const float * mlx_array_data_float32(const mlx_array arr) {
    return mlx_array_data_float32_(arr);
}

const double * mlx_array_data_float64(const mlx_array arr) {
    return mlx_array_data_float64_(arr);
}

const float _Complex * mlx_array_data_complex64(const mlx_array arr) {
    return mlx_array_data_complex64_(arr);
}

const float16_t * mlx_array_data_float16(const mlx_array arr) {
    return mlx_array_data_float16_(arr);
}

const bfloat16_t * mlx_array_data_bfloat16(const mlx_array arr) {
    return mlx_array_data_bfloat16_(arr);
}

int _mlx_array_is_available(bool* res, const mlx_array arr) {
    return _mlx_array_is_available_(res, arr);
}

int _mlx_array_wait(const mlx_array arr) {
    return _mlx_array_wait_(arr);
}

int _mlx_array_is_contiguous(bool* res, const mlx_array arr) {
    return _mlx_array_is_contiguous_(res, arr);
}

int _mlx_array_is_row_contiguous(bool* res, const mlx_array arr) {
    return _mlx_array_is_row_contiguous_(res, arr);
}

int _mlx_array_is_col_contiguous(bool* res, const mlx_array arr) {
    return _mlx_array_is_col_contiguous_(res, arr);
}

mlx_closure mlx_closure_new(void) {
    return mlx_closure_new_();
}

int mlx_closure_free(mlx_closure cls) {
    return mlx_closure_free_(cls);
}

mlx_closure mlx_closure_new_func(
    int (*fun)(mlx_vector_array*, const mlx_vector_array)) {
    return mlx_closure_new_func_(fun);
}

mlx_closure mlx_closure_new_func_payload(
    int (*fun)(mlx_vector_array*, const mlx_vector_array, void*),
    void* payload,
    void (*dtor)(void*)) {
    return mlx_closure_new_func_payload_(fun, payload, dtor);
}

int mlx_closure_set(mlx_closure* cls, const mlx_closure src) {
    return mlx_closure_set_(cls, src);
}

int mlx_closure_apply(
    mlx_vector_array* res,
    mlx_closure cls,
    const mlx_vector_array input) {
    return mlx_closure_apply_(res, cls, input);
}

mlx_closure mlx_closure_new_unary(int (*fun)(mlx_array*, const mlx_array)) {
    return mlx_closure_new_unary_(fun);
}

mlx_closure_kwargs mlx_closure_kwargs_new(void) {
    return mlx_closure_kwargs_new_();
}

int mlx_closure_kwargs_free(mlx_closure_kwargs cls) {
    return mlx_closure_kwargs_free_(cls);
}

mlx_closure_kwargs mlx_closure_kwargs_new_func(int (*fun)(
    mlx_vector_array*,
    const mlx_vector_array,
    const mlx_map_string_to_array)) {
    return mlx_closure_kwargs_new_func_(fun);
}

mlx_closure_kwargs mlx_closure_kwargs_new_func_payload(
    int (*fun)(
        mlx_vector_array*,
        const mlx_vector_array,
        const mlx_map_string_to_array,
        void*),
    void* payload,
    void (*dtor)(void*)) {
    return mlx_closure_kwargs_new_func_payload_(fun, payload, dtor);
}

int mlx_closure_kwargs_set(
    mlx_closure_kwargs* cls,
    const mlx_closure_kwargs src) {
    return mlx_closure_kwargs_set_(cls, src);
}

int mlx_closure_kwargs_apply(
    mlx_vector_array* res,
    mlx_closure_kwargs cls,
    const mlx_vector_array input_0,
    const mlx_map_string_to_array input_1) {
    return mlx_closure_kwargs_apply_(res, cls, input_0, input_1);
}

mlx_closure_value_and_grad mlx_closure_value_and_grad_new(void) {
    return mlx_closure_value_and_grad_new_();
}

int mlx_closure_value_and_grad_free(mlx_closure_value_and_grad cls) {
    return mlx_closure_value_and_grad_free_(cls);
}

mlx_closure_value_and_grad mlx_closure_value_and_grad_new_func(
    int (*fun)(mlx_vector_array*, mlx_vector_array*, const mlx_vector_array)) {
    return mlx_closure_value_and_grad_new_func_(fun);
}

mlx_closure_value_and_grad mlx_closure_value_and_grad_new_func_payload(
    int (*fun)(
        mlx_vector_array*,
        mlx_vector_array*,
        const mlx_vector_array,
        void*),
    void* payload,
    void (*dtor)(void*)) {
    return mlx_closure_value_and_grad_new_func_payload_(fun, payload, dtor);
}

int mlx_closure_value_and_grad_set(
    mlx_closure_value_and_grad* cls,
    const mlx_closure_value_and_grad src) {
    return mlx_closure_value_and_grad_set_(cls, src);
}

int mlx_closure_value_and_grad_apply(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    mlx_closure_value_and_grad cls,
    const mlx_vector_array input) {
    return mlx_closure_value_and_grad_apply_(res_0, res_1, cls, input);
}

mlx_closure_custom mlx_closure_custom_new(void) {
    return mlx_closure_custom_new_();
}

int mlx_closure_custom_free(mlx_closure_custom cls) {
    return mlx_closure_custom_free_(cls);
}

mlx_closure_custom mlx_closure_custom_new_func(int (*fun)(
    mlx_vector_array*,
    const mlx_vector_array,
    const mlx_vector_array,
    const mlx_vector_array)) {
    return mlx_closure_custom_new_func_(fun);
}

mlx_closure_custom mlx_closure_custom_new_func_payload(
    int (*fun)(
        mlx_vector_array*,
        const mlx_vector_array,
        const mlx_vector_array,
        const mlx_vector_array,
        void*),
    void* payload,
    void (*dtor)(void*)) {
    return mlx_closure_custom_new_func_payload_(fun, payload, dtor);
}

int mlx_closure_custom_set(
    mlx_closure_custom* cls,
    const mlx_closure_custom src) {
    return mlx_closure_custom_set_(cls, src);
}

int mlx_closure_custom_apply(
    mlx_vector_array* res,
    mlx_closure_custom cls,
    const mlx_vector_array input_0,
    const mlx_vector_array input_1,
    const mlx_vector_array input_2) {
    return mlx_closure_custom_apply_(res, cls, input_0, input_1, input_2);
}

mlx_closure_custom_jvp mlx_closure_custom_jvp_new(void) {
    return mlx_closure_custom_jvp_new_();
}

int mlx_closure_custom_jvp_free(mlx_closure_custom_jvp cls) {
    return mlx_closure_custom_jvp_free_(cls);
}

mlx_closure_custom_jvp mlx_closure_custom_jvp_new_func(int (*fun)(
    mlx_vector_array*,
    const mlx_vector_array,
    const mlx_vector_array,
    const int*,
    size_t _num)) {
    return mlx_closure_custom_jvp_new_func_(fun);
}

mlx_closure_custom_jvp mlx_closure_custom_jvp_new_func_payload(
    int (*fun)(
        mlx_vector_array*,
        const mlx_vector_array,
        const mlx_vector_array,
        const int*,
        size_t _num,
        void*),
    void* payload,
    void (*dtor)(void*)) {
    return mlx_closure_custom_jvp_new_func_payload_(fun, payload, dtor);
}

int mlx_closure_custom_jvp_set(
    mlx_closure_custom_jvp* cls,
    const mlx_closure_custom_jvp src) {
    return mlx_closure_custom_jvp_set_(cls, src);
}

int mlx_closure_custom_jvp_apply(
    mlx_vector_array* res,
    mlx_closure_custom_jvp cls,
    const mlx_vector_array input_0,
    const mlx_vector_array input_1,
    const int* input_2,
    size_t input_2_num) {
    return mlx_closure_custom_jvp_apply_(res, cls, input_0, input_1, input_2, input_2_num);
}

mlx_closure_custom_vmap mlx_closure_custom_vmap_new(void) {
    return mlx_closure_custom_vmap_new_();
}

int mlx_closure_custom_vmap_free(mlx_closure_custom_vmap cls) {
    return mlx_closure_custom_vmap_free_(cls);
}

mlx_closure_custom_vmap mlx_closure_custom_vmap_new_func(int (*fun)(
    mlx_vector_array*,
    mlx_vector_int*,
    const mlx_vector_array,
    const int*,
    size_t _num)) {
    return mlx_closure_custom_vmap_new_func_(fun);
}

mlx_closure_custom_vmap mlx_closure_custom_vmap_new_func_payload(
    int (*fun)(
        mlx_vector_array*,
        mlx_vector_int*,
        const mlx_vector_array,
        const int*,
        size_t _num,
        void*),
    void* payload,
    void (*dtor)(void*)) {
    return mlx_closure_custom_vmap_new_func_payload_(fun, payload, dtor);
}

int mlx_closure_custom_vmap_set(
    mlx_closure_custom_vmap* cls,
    const mlx_closure_custom_vmap src) {
    return mlx_closure_custom_vmap_set_(cls, src);
}

int mlx_closure_custom_vmap_apply(
    mlx_vector_array* res_0,
    mlx_vector_int* res_1,
    mlx_closure_custom_vmap cls,
    const mlx_vector_array input_0,
    const int* input_1,
    size_t input_1_num) {
    return mlx_closure_custom_vmap_apply_(res_0, res_1, cls, input_0, input_1, input_1_num);
}

int mlx_compile(mlx_closure* res, const mlx_closure fun, bool shapeless) {
    return mlx_compile_(res, fun, shapeless);
}

int mlx_detail_compile(
    mlx_closure* res,
    const mlx_closure fun,
    uintptr_t fun_id,
    bool shapeless,
    const uint64_t* constants,
    size_t constants_num) {
    return mlx_detail_compile_(res, fun, fun_id, shapeless, constants, constants_num);
}

int mlx_detail_compile_clear_cache(void) {
    return mlx_detail_compile_clear_cache_();
}

int mlx_detail_compile_erase(uintptr_t fun_id) {
    return mlx_detail_compile_erase_(fun_id);
}

int mlx_disable_compile(void) {
    return mlx_disable_compile_();
}

int mlx_enable_compile(void) {
    return mlx_enable_compile_();
}

int mlx_set_compile_mode(mlx_compile_mode mode) {
    return mlx_set_compile_mode_(mode);
}

mlx_device mlx_device_new(void) {
    return mlx_device_new_();
}

mlx_device mlx_device_new_type(mlx_device_type type, int index) {
    return mlx_device_new_type_(type, index);
}

int mlx_device_free(mlx_device dev) {
    return mlx_device_free_(dev);
}

int mlx_device_set(mlx_device* dev, const mlx_device src) {
    return mlx_device_set_(dev, src);
}

int mlx_device_tostring(mlx_string* str, mlx_device dev) {
    return mlx_device_tostring_(str, dev);
}

bool mlx_device_equal(mlx_device lhs, mlx_device rhs) {
    return mlx_device_equal_(lhs, rhs);
}

int mlx_device_get_index(int* index, mlx_device dev) {
    return mlx_device_get_index_(index, dev);
}

int mlx_device_get_type(mlx_device_type* type, mlx_device dev) {
    return mlx_device_get_type_(type, dev);
}

int mlx_get_default_device(mlx_device* dev) {
    return mlx_get_default_device_(dev);
}

int mlx_set_default_device(mlx_device dev) {
    return mlx_set_default_device_(dev);
}

int mlx_distributed_group_rank(mlx_distributed_group group) {
    return mlx_distributed_group_rank_(group);
}

int mlx_distributed_group_size(mlx_distributed_group group) {
    return mlx_distributed_group_size_(group);
}

mlx_distributed_group mlx_distributed_group_split(mlx_distributed_group group, int color, int key) {
    return mlx_distributed_group_split_(group, color, key);
}

bool mlx_distributed_is_available(void) {
    return mlx_distributed_is_available_();
}

mlx_distributed_group mlx_distributed_init(bool strict) {
    return mlx_distributed_init_(strict);
}

int mlx_distributed_all_gather(
    mlx_array* res,
    const mlx_array x,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream S) {
    return mlx_distributed_all_gather_(res, x, group, S);
}

int mlx_distributed_all_max(
    mlx_array* res,
    const mlx_array x,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s) {
    return mlx_distributed_all_max_(res, x, group, s);
}

int mlx_distributed_all_min(
    mlx_array* res,
    const mlx_array x,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s) {
    return mlx_distributed_all_min_(res, x, group, s);
}

int mlx_distributed_all_sum(
    mlx_array* res,
    const mlx_array x,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s) {
    return mlx_distributed_all_sum_(res, x, group, s);
}

int mlx_distributed_recv(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    int src,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s) {
    return mlx_distributed_recv_(res, shape, shape_num, dtype, src, group, s);
}

int mlx_distributed_recv_like(
    mlx_array* res,
    const mlx_array x,
    int src,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s) {
    return mlx_distributed_recv_like_(res, x, src, group, s);
}

int mlx_distributed_send(
    mlx_array* res,
    const mlx_array x,
    int dst,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s) {
    return mlx_distributed_send_(res, x, dst, group, s);
}

int mlx_distributed_sum_scatter(
    mlx_array* res,
    const mlx_array x,
    const mlx_distributed_group group /* may be null */,
    const mlx_stream s) {
    return mlx_distributed_sum_scatter_(res, x, group, s);
}

void mlx_set_error_handler(
    mlx_error_handler_func handler,
    void* data,
    void (*dtor)(void*)) {
    return mlx_set_error_handler_(handler, data, dtor);
}

void _mlx_error(const char* file, const int line, const char* fmt, ...) {
    return _mlx_error_(file, line, fmt);
}

int mlx_export_function(
    const char* file,
    const mlx_closure fun,
    const mlx_vector_array args,
    bool shapeless) {
    return mlx_export_function_(file, fun, args, shapeless);
}

int mlx_export_function_kwargs(
    const char* file,
    const mlx_closure_kwargs fun,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs,
    bool shapeless) {
    return mlx_export_function_kwargs_(file, fun, args, kwargs, shapeless);
}

mlx_function_exporter mlx_function_exporter_new(
    const char* file,
    const mlx_closure fun,
    bool shapeless) {
    return mlx_function_exporter_new_(file, fun, shapeless);
}

int mlx_function_exporter_free(mlx_function_exporter xfunc) {
    return mlx_function_exporter_free_(xfunc);
}

int mlx_function_exporter_apply(
    const mlx_function_exporter xfunc,
    const mlx_vector_array args) {
    return mlx_function_exporter_apply_(xfunc, args);
}

int mlx_function_exporter_apply_kwargs(
    const mlx_function_exporter xfunc,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs) {
    return mlx_function_exporter_apply_kwargs_(xfunc, args, kwargs);
}

mlx_imported_function mlx_imported_function_new(const char* file) {
    return mlx_imported_function_new_(file);
}

int mlx_imported_function_free(mlx_imported_function xfunc) {
    return mlx_imported_function_free_(xfunc);
}

int mlx_imported_function_apply(
    mlx_vector_array* res,
    const mlx_imported_function xfunc,
    const mlx_vector_array args) {
    return mlx_imported_function_apply_(res, xfunc, args);
}

int mlx_imported_function_apply_kwargs(
    mlx_vector_array* res,
    const mlx_imported_function xfunc,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs) {
    return mlx_imported_function_apply_kwargs_(res, xfunc, args, kwargs);
}

mlx_fast_cuda_kernel_config mlx_fast_cuda_kernel_config_new(void) {
    return mlx_fast_cuda_kernel_config_new_();
}

void mlx_fast_cuda_kernel_config_free(mlx_fast_cuda_kernel_config cls) {
    return mlx_fast_cuda_kernel_config_free_(cls);
}

int mlx_fast_cuda_kernel_config_add_output_arg(
    mlx_fast_cuda_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype) {
    return mlx_fast_cuda_kernel_config_add_output_arg_(cls, shape, size, dtype);
}

int mlx_fast_cuda_kernel_config_set_grid(
    mlx_fast_cuda_kernel_config cls,
    int grid1,
    int grid2,
    int grid3) {
    return mlx_fast_cuda_kernel_config_set_grid_(cls, grid1, grid2, grid3);
}

int mlx_fast_cuda_kernel_config_set_thread_group(
    mlx_fast_cuda_kernel_config cls,
    int thread1,
    int thread2,
    int thread3) {
    return mlx_fast_cuda_kernel_config_set_thread_group_(cls, thread1, thread2, thread3);
}

int mlx_fast_cuda_kernel_config_set_init_value(
    mlx_fast_cuda_kernel_config cls,
    float value) {
    return mlx_fast_cuda_kernel_config_set_init_value_(cls, value);
}

int mlx_fast_cuda_kernel_config_set_verbose(
    mlx_fast_cuda_kernel_config cls,
    bool verbose) {
    return mlx_fast_cuda_kernel_config_set_verbose_(cls, verbose);
}

int mlx_fast_cuda_kernel_config_add_template_arg_dtype(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    mlx_dtype dtype) {
    return mlx_fast_cuda_kernel_config_add_template_arg_dtype_(cls, name, dtype);
}

int mlx_fast_cuda_kernel_config_add_template_arg_int(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    int value) {
    return mlx_fast_cuda_kernel_config_add_template_arg_int_(cls, name, value);
}

int mlx_fast_cuda_kernel_config_add_template_arg_bool(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    bool value) {
    return mlx_fast_cuda_kernel_config_add_template_arg_bool_(cls, name, value);
}

mlx_fast_cuda_kernel mlx_fast_cuda_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    int shared_memory) {
    return mlx_fast_cuda_kernel_new_(name, input_names, output_names, source, header, ensure_row_contiguous, shared_memory);
}

void mlx_fast_cuda_kernel_free(mlx_fast_cuda_kernel cls) {
    return mlx_fast_cuda_kernel_free_(cls);
}

int mlx_fast_cuda_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_cuda_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_cuda_kernel_config config,
    const mlx_stream stream) {
    return mlx_fast_cuda_kernel_apply_(outputs, cls, inputs, config, stream);
}

int mlx_fast_layer_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    const mlx_array bias /* may be null */,
    float eps,
    const mlx_stream s) {
    return mlx_fast_layer_norm_(res, x, weight, bias, eps, s);
}

mlx_fast_metal_kernel_config mlx_fast_metal_kernel_config_new(void) {
    return mlx_fast_metal_kernel_config_new_();
}

void mlx_fast_metal_kernel_config_free(mlx_fast_metal_kernel_config cls) {
    return mlx_fast_metal_kernel_config_free_(cls);
}

int mlx_fast_metal_kernel_config_add_output_arg(
    mlx_fast_metal_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype) {
    return mlx_fast_metal_kernel_config_add_output_arg_(cls, shape, size, dtype);
}

int mlx_fast_metal_kernel_config_set_grid(
    mlx_fast_metal_kernel_config cls,
    int grid1,
    int grid2,
    int grid3) {
    return mlx_fast_metal_kernel_config_set_grid_(cls, grid1, grid2, grid3);
}

int mlx_fast_metal_kernel_config_set_thread_group(
    mlx_fast_metal_kernel_config cls,
    int thread1,
    int thread2,
    int thread3) {
    return mlx_fast_metal_kernel_config_set_thread_group_(cls, thread1, thread2, thread3);
}

int mlx_fast_metal_kernel_config_set_init_value(
    mlx_fast_metal_kernel_config cls,
    float value) {
    return mlx_fast_metal_kernel_config_set_init_value_(cls, value);
}

int mlx_fast_metal_kernel_config_set_verbose(
    mlx_fast_metal_kernel_config cls,
    bool verbose) {
    return mlx_fast_metal_kernel_config_set_verbose_(cls, verbose);
}

int mlx_fast_metal_kernel_config_add_template_arg_dtype(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    mlx_dtype dtype) {
    return mlx_fast_metal_kernel_config_add_template_arg_dtype_(cls, name, dtype);
}

int mlx_fast_metal_kernel_config_add_template_arg_int(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    int value) {
    return mlx_fast_metal_kernel_config_add_template_arg_int_(cls, name, value);
}

int mlx_fast_metal_kernel_config_add_template_arg_bool(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    bool value) {
    return mlx_fast_metal_kernel_config_add_template_arg_bool_(cls, name, value);
}

mlx_fast_metal_kernel mlx_fast_metal_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    bool atomic_outputs) {
    return mlx_fast_metal_kernel_new_(name, input_names, output_names, source, header, ensure_row_contiguous, atomic_outputs);
}

void mlx_fast_metal_kernel_free(mlx_fast_metal_kernel cls) {
    return mlx_fast_metal_kernel_free_(cls);
}

int mlx_fast_metal_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_metal_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_metal_kernel_config config,
    const mlx_stream stream) {
    return mlx_fast_metal_kernel_apply_(outputs, cls, inputs, config, stream);
}

int mlx_fast_rms_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    float eps,
    const mlx_stream s) {
    return mlx_fast_rms_norm_(res, x, weight, eps, s);
}

int mlx_fast_rope(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    int offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s) {
    return mlx_fast_rope_(res, x, dims, traditional, base, scale, offset, freqs, s);
}

int mlx_fast_scaled_dot_product_attention(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array keys,
    const mlx_array values,
    float scale,
    const char* mask_mode,
    const mlx_array mask_arr /* may be null */,
    const mlx_array sinks /* may be null */,
    const mlx_stream s) {
    return mlx_fast_scaled_dot_product_attention_(res, queries, keys, values, scale, mask_mode, mask_arr, sinks, s);
}

int mlx_fft_fft(
    mlx_array* res,
    const mlx_array a,
    int n,
    int axis,
    const mlx_stream s) {
    return mlx_fft_fft_(res, a, n, axis, s);
}

int mlx_fft_fft2(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
    return mlx_fft_fft2_(res, a, n, n_num, axes, axes_num, s);
}

int mlx_fft_fftn(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
    return mlx_fft_fftn_(res, a, n, n_num, axes, axes_num, s);
}

int mlx_fft_fftshift(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
    return mlx_fft_fftshift_(res, a, axes, axes_num, s);
}

int mlx_fft_ifft(
    mlx_array* res,
    const mlx_array a,
    int n,
    int axis,
    const mlx_stream s) {
    return mlx_fft_ifft_(res, a, n, axis, s);
}

int mlx_fft_ifft2(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
    return mlx_fft_ifft2_(res, a, n, n_num, axes, axes_num, s);
}

int mlx_fft_ifftn(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
    return mlx_fft_ifftn_(res, a, n, n_num, axes, axes_num, s);
}

int mlx_fft_ifftshift(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
    return mlx_fft_ifftshift_(res, a, axes, axes_num, s);
}

int mlx_fft_irfft(
    mlx_array* res,
    const mlx_array a,
    int n,
    int axis,
    const mlx_stream s) {
    return mlx_fft_irfft_(res, a, n, axis, s);
}

int mlx_fft_irfft2(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
    return mlx_fft_irfft2_(res, a, n, n_num, axes, axes_num, s);
}

int mlx_fft_irfftn(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
    return mlx_fft_irfftn_(res, a, n, n_num, axes, axes_num, s);
}

int mlx_fft_rfft(
    mlx_array* res,
    const mlx_array a,
    int n,
    int axis,
    const mlx_stream s) {
    return mlx_fft_rfft_(res, a, n, axis, s);
}

int mlx_fft_rfft2(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
    return mlx_fft_rfft2_(res, a, n, n_num, axes, axes_num, s);
}

int mlx_fft_rfftn(
    mlx_array* res,
    const mlx_array a,
    const int* n,
    size_t n_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
    return mlx_fft_rfftn_(res, a, n, n_num, axes, axes_num, s);
}

mlx_io_reader mlx_io_reader_new(void* desc, mlx_io_vtable vtable) {
    return mlx_io_reader_new_(desc, vtable);
}

int mlx_io_reader_descriptor(void** desc_, mlx_io_reader io) {
    return mlx_io_reader_descriptor_(desc_, io);
}

int mlx_io_reader_tostring(mlx_string* str_, mlx_io_reader io) {
    return mlx_io_reader_tostring_(str_, io);
}

int mlx_io_reader_free(mlx_io_reader io) {
    return mlx_io_reader_free_(io);
}

mlx_io_writer mlx_io_writer_new(void* desc, mlx_io_vtable vtable) {
    return mlx_io_writer_new_(desc, vtable);
}

int mlx_io_writer_descriptor(void** desc_, mlx_io_writer io) {
    return mlx_io_writer_descriptor_(desc_, io);
}

int mlx_io_writer_tostring(mlx_string* str_, mlx_io_writer io) {
    return mlx_io_writer_tostring_(str_, io);
}

int mlx_io_writer_free(mlx_io_writer io) {
    return mlx_io_writer_free_(io);
}

int mlx_load_reader(
    mlx_array* res,
    mlx_io_reader in_stream,
    const mlx_stream s) {
    return mlx_load_reader_(res, in_stream, s);
}

int mlx_load(mlx_array* res, const char* file, const mlx_stream s) {
    return mlx_load_(res, file, s);
}

int mlx_load_safetensors_reader(
    mlx_map_string_to_array* res_0,
    mlx_map_string_to_string* res_1,
    mlx_io_reader in_stream,
    const mlx_stream s) {
    return mlx_load_safetensors_reader_(res_0, res_1, in_stream, s);
}

int mlx_load_safetensors(
    mlx_map_string_to_array* res_0,
    mlx_map_string_to_string* res_1,
    const char* file,
    const mlx_stream s) {
    return mlx_load_safetensors_(res_0, res_1, file, s);
}

int mlx_save_writer(mlx_io_writer out_stream, const mlx_array a) {
    return mlx_save_writer_(out_stream, a);
}

int mlx_save(const char* file, const mlx_array a) {
    return mlx_save_(file, a);
}

int mlx_save_safetensors_writer(
    mlx_io_writer in_stream,
    const mlx_map_string_to_array param,
    const mlx_map_string_to_string metadata) {
    return mlx_save_safetensors_writer_(in_stream, param, metadata);
}

int mlx_save_safetensors(
    const char* file,
    const mlx_map_string_to_array param,
    const mlx_map_string_to_string metadata) {
    return mlx_save_safetensors_(file, param, metadata);
}

int mlx_linalg_cholesky(
    mlx_array* res,
    const mlx_array a,
    bool upper,
    const mlx_stream s) {
    return mlx_linalg_cholesky_(res, a, upper, s);
}

int mlx_linalg_cholesky_inv(
    mlx_array* res,
    const mlx_array a,
    bool upper,
    const mlx_stream s) {
    return mlx_linalg_cholesky_inv_(res, a, upper, s);
}

int mlx_linalg_cross(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    int axis,
    const mlx_stream s) {
    return mlx_linalg_cross_(res, a, b, axis, s);
}

int mlx_linalg_eig(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const mlx_stream s) {
    return mlx_linalg_eig_(res_0, res_1, a, s);
}

int mlx_linalg_eigh(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const char* UPLO,
    const mlx_stream s) {
    return mlx_linalg_eigh_(res_0, res_1, a, UPLO, s);
}

int mlx_linalg_eigvals(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_linalg_eigvals_(res, a, s);
}

int mlx_linalg_eigvalsh(
    mlx_array* res,
    const mlx_array a,
    const char* UPLO,
    const mlx_stream s) {
    return mlx_linalg_eigvalsh_(res, a, UPLO, s);
}

int mlx_linalg_inv(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_linalg_inv_(res, a, s);
}

int mlx_linalg_lu(mlx_vector_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_linalg_lu_(res, a, s);
}

int mlx_linalg_lu_factor(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const mlx_stream s) {
    return mlx_linalg_lu_factor_(res_0, res_1, a, s);
}

int mlx_linalg_norm(
    mlx_array* res,
    const mlx_array a,
    double ord,
    const int* axis /* may be null */,
    size_t axis_num,
    bool keepdims,
    const mlx_stream s) {
    return mlx_linalg_norm_(res, a, ord, axis, axis_num, keepdims, s);
}

int mlx_linalg_norm_matrix(
    mlx_array* res,
    const mlx_array a,
    const char* ord,
    const int* axis /* may be null */,
    size_t axis_num,
    bool keepdims,
    const mlx_stream s) {
    return mlx_linalg_norm_matrix_(res, a, ord, axis, axis_num, keepdims, s);
}

int mlx_linalg_norm_l2(
    mlx_array* res,
    const mlx_array a,
    const int* axis /* may be null */,
    size_t axis_num,
    bool keepdims,
    const mlx_stream s) {
    return mlx_linalg_norm_l2_(res, a, axis, axis_num, keepdims, s);
}

int mlx_linalg_pinv(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_linalg_pinv_(res, a, s);
}

int mlx_linalg_qr(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array a,
    const mlx_stream s) {
    return mlx_linalg_qr_(res_0, res_1, a, s);
}

int mlx_linalg_solve(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_linalg_solve_(res, a, b, s);
}

int mlx_linalg_solve_triangular(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    bool upper,
    const mlx_stream s) {
    return mlx_linalg_solve_triangular_(res, a, b, upper, s);
}

int mlx_linalg_svd(
    mlx_vector_array* res,
    const mlx_array a,
    bool compute_uv,
    const mlx_stream s) {
    return mlx_linalg_svd_(res, a, compute_uv, s);
}

int mlx_linalg_tri_inv(
    mlx_array* res,
    const mlx_array a,
    bool upper,
    const mlx_stream s) {
    return mlx_linalg_tri_inv_(res, a, upper, s);
}

mlx_map_string_to_array mlx_map_string_to_array_new(void) {
    return mlx_map_string_to_array_new_();
}

int mlx_map_string_to_array_set(
    mlx_map_string_to_array* map,
    const mlx_map_string_to_array src) {
    return mlx_map_string_to_array_set_(map, src);
}

int mlx_map_string_to_array_free(mlx_map_string_to_array map) {
    return mlx_map_string_to_array_free_(map);
}

int mlx_map_string_to_array_insert(
    mlx_map_string_to_array map,
    const char* key,
    const mlx_array value) {
    return mlx_map_string_to_array_insert_(map, key, value);
}

int mlx_map_string_to_array_get(
    mlx_array* value,
    const mlx_map_string_to_array map,
    const char* key) {
    return mlx_map_string_to_array_get_(value, map, key);
}

mlx_map_string_to_array_iterator mlx_map_string_to_array_iterator_new(
    mlx_map_string_to_array map) {
    return mlx_map_string_to_array_iterator_new_(map);
}

int mlx_map_string_to_array_iterator_free(mlx_map_string_to_array_iterator it) {
    return mlx_map_string_to_array_iterator_free_(it);
}

int mlx_map_string_to_array_iterator_next(
    const char** key,
    mlx_array* value,
    mlx_map_string_to_array_iterator it) {
    return mlx_map_string_to_array_iterator_next_(key, value, it);
}

mlx_map_string_to_string mlx_map_string_to_string_new(void) {
    return mlx_map_string_to_string_new_();
}

int mlx_map_string_to_string_set(
    mlx_map_string_to_string* map,
    const mlx_map_string_to_string src) {
    return mlx_map_string_to_string_set_(map, src);
}

int mlx_map_string_to_string_free(mlx_map_string_to_string map) {
    return mlx_map_string_to_string_free_(map);
}

int mlx_map_string_to_string_insert(
    mlx_map_string_to_string map,
    const char* key,
    const char* value) {
    return mlx_map_string_to_string_insert_(map, key, value);
}

int mlx_map_string_to_string_get(
    const char** value,
    const mlx_map_string_to_string map,
    const char* key) {
    return mlx_map_string_to_string_get_(value, map, key);
}

mlx_map_string_to_string_iterator mlx_map_string_to_string_iterator_new(
    mlx_map_string_to_string map) {
    return mlx_map_string_to_string_iterator_new_(map);
}

int mlx_map_string_to_string_iterator_free(
    mlx_map_string_to_string_iterator it) {
    return mlx_map_string_to_string_iterator_free_(it);
}

int mlx_map_string_to_string_iterator_next(
    const char** key,
    const char** value,
    mlx_map_string_to_string_iterator it) {
    return mlx_map_string_to_string_iterator_next_(key, value, it);
}

int mlx_clear_cache(void) {
    return mlx_clear_cache_();
}

int mlx_get_active_memory(size_t* res) {
    return mlx_get_active_memory_(res);
}

int mlx_get_cache_memory(size_t* res) {
    return mlx_get_cache_memory_(res);
}

int mlx_get_memory_limit(size_t* res) {
    return mlx_get_memory_limit_(res);
}

int mlx_get_peak_memory(size_t* res) {
    return mlx_get_peak_memory_(res);
}

int mlx_reset_peak_memory(void) {
    return mlx_reset_peak_memory_();
}

int mlx_set_cache_limit(size_t* res, size_t limit) {
    return mlx_set_cache_limit_(res, limit);
}

int mlx_set_memory_limit(size_t* res, size_t limit) {
    return mlx_set_memory_limit_(res, limit);
}

int mlx_set_wired_limit(size_t* res, size_t limit) {
    return mlx_set_wired_limit_(res, limit);
}

mlx_metal_device_info_t mlx_metal_device_info(void) {
    return mlx_metal_device_info_();
}

int mlx_metal_is_available(bool* res) {
    return mlx_metal_is_available_(res);
}

int mlx_metal_start_capture(const char* path) {
    return mlx_metal_start_capture_(path);
}

int mlx_metal_stop_capture(void) {
    return mlx_metal_stop_capture_();
}

int mlx_abs(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_abs_(res, a, s);
}

int mlx_add(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_add_(res, a, b, s);
}

int mlx_addmm(
    mlx_array* res,
    const mlx_array c,
    const mlx_array a,
    const mlx_array b,
    float alpha,
    float beta,
    const mlx_stream s) {
    return mlx_addmm_(res, c, a, b, alpha, beta, s);
}

int mlx_all_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s) {
    return mlx_all_axes_(res, a, axes, axes_num, keepdims, s);
}

int mlx_all_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s) {
    return mlx_all_axis_(res, a, axis, keepdims, s);
}

int mlx_all(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s) {
    return mlx_all_(res, a, keepdims, s);
}

int mlx_allclose(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    double rtol,
    double atol,
    bool equal_nan,
    const mlx_stream s) {
    return mlx_allclose_(res, a, b, rtol, atol, equal_nan, s);
}

int mlx_any_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s) {
    return mlx_any_axes_(res, a, axes, axes_num, keepdims, s);
}

int mlx_any_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s) {
    return mlx_any_axis_(res, a, axis, keepdims, s);
}

int mlx_any(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s) {
    return mlx_any_(res, a, keepdims, s);
}

int mlx_arange(
    mlx_array* res,
    double start,
    double stop,
    double step,
    mlx_dtype dtype,
    const mlx_stream s) {
    return mlx_arange_(res, start, stop, step, dtype, s);
}

int mlx_arccos(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_arccos_(res, a, s);
}

int mlx_arccosh(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_arccosh_(res, a, s);
}

int mlx_arcsin(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_arcsin_(res, a, s);
}

int mlx_arcsinh(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_arcsinh_(res, a, s);
}

int mlx_arctan(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_arctan_(res, a, s);
}

int mlx_arctan2(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_arctan2_(res, a, b, s);
}

int mlx_arctanh(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_arctanh_(res, a, s);
}

int mlx_argmax_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s) {
    return mlx_argmax_axis_(res, a, axis, keepdims, s);
}

int mlx_argmax(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s) {
    return mlx_argmax_(res, a, keepdims, s);
}

int mlx_argmin_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s) {
    return mlx_argmin_axis_(res, a, axis, keepdims, s);
}

int mlx_argmin(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s) {
    return mlx_argmin_(res, a, keepdims, s);
}

int mlx_argpartition_axis(
    mlx_array* res,
    const mlx_array a,
    int kth,
    int axis,
    const mlx_stream s) {
    return mlx_argpartition_axis_(res, a, kth, axis, s);
}

int mlx_argpartition(
    mlx_array* res,
    const mlx_array a,
    int kth,
    const mlx_stream s) {
    return mlx_argpartition_(res, a, kth, s);
}

int mlx_argsort_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    const mlx_stream s) {
    return mlx_argsort_axis_(res, a, axis, s);
}

int mlx_argsort(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_argsort_(res, a, s);
}

int mlx_array_equal(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    bool equal_nan,
    const mlx_stream s) {
    return mlx_array_equal_(res, a, b, equal_nan, s);
}

int mlx_as_strided(
    mlx_array* res,
    const mlx_array a,
    const int* shape,
    size_t shape_num,
    const int64_t* strides,
    size_t strides_num,
    size_t offset,
    const mlx_stream s) {
    return mlx_as_strided_(res, a, shape, shape_num, strides, strides_num, offset, s);
}

int mlx_astype(
    mlx_array* res,
    const mlx_array a,
    mlx_dtype dtype,
    const mlx_stream s) {
    return mlx_astype_(res, a, dtype, s);
}

int mlx_atleast_1d(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_atleast_1d_(res, a, s);
}

int mlx_atleast_2d(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_atleast_2d_(res, a, s);
}

int mlx_atleast_3d(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_atleast_3d_(res, a, s);
}

int mlx_bitwise_and(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_bitwise_and_(res, a, b, s);
}

int mlx_bitwise_invert(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_bitwise_invert_(res, a, s);
}

int mlx_bitwise_or(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_bitwise_or_(res, a, b, s);
}

int mlx_bitwise_xor(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_bitwise_xor_(res, a, b, s);
}

int mlx_block_masked_mm(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    int block_size,
    const mlx_array mask_out /* may be null */,
    const mlx_array mask_lhs /* may be null */,
    const mlx_array mask_rhs /* may be null */,
    const mlx_stream s) {
    return mlx_block_masked_mm_(res, a, b, block_size, mask_out, mask_lhs, mask_rhs, s);
}

int mlx_broadcast_arrays(
    mlx_vector_array* res,
    const mlx_vector_array inputs,
    const mlx_stream s) {
    return mlx_broadcast_arrays_(res, inputs, s);
}

int mlx_broadcast_to(
    mlx_array* res,
    const mlx_array a,
    const int* shape,
    size_t shape_num,
    const mlx_stream s) {
    return mlx_broadcast_to_(res, a, shape, shape_num, s);
}

int mlx_ceil(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_ceil_(res, a, s);
}

int mlx_clip(
    mlx_array* res,
    const mlx_array a,
    const mlx_array a_min /* may be null */,
    const mlx_array a_max /* may be null */,
    const mlx_stream s) {
    return mlx_clip_(res, a, a_min, a_max, s);
}

int mlx_concatenate_axis(
    mlx_array* res,
    const mlx_vector_array arrays,
    int axis,
    const mlx_stream s) {
    return mlx_concatenate_axis_(res, arrays, axis, s);
}

int mlx_concatenate(
    mlx_array* res,
    const mlx_vector_array arrays,
    const mlx_stream s) {
    return mlx_concatenate_(res, arrays, s);
}

int mlx_conjugate(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_conjugate_(res, a, s);
}

int mlx_contiguous(
    mlx_array* res,
    const mlx_array a,
    bool allow_col_major,
    const mlx_stream s) {
    return mlx_contiguous_(res, a, allow_col_major, s);
}

int mlx_conv1d(
    mlx_array* res,
    const mlx_array input,
    const mlx_array weight,
    int stride,
    int padding,
    int dilation,
    int groups,
    const mlx_stream s) {
    return mlx_conv1d_(res, input, weight, stride, padding, dilation, groups, s);
}

int mlx_conv2d(
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
    const mlx_stream s) {
    return mlx_conv2d_(res, input, weight, stride_0, stride_1, padding_0, padding_1, dilation_0, dilation_1, groups, s);
}

int mlx_conv3d(
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
    const mlx_stream s) {
    return mlx_conv3d_(res, input, weight, stride_0, stride_1, stride_2, padding_0, padding_1, padding_2, dilation_0, dilation_1, dilation_2, groups, s);
}

int mlx_conv_general(
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
    const mlx_stream s) {
    return mlx_conv_general_(res, input, weight, stride, stride_num, padding_lo, padding_lo_num, padding_hi, padding_hi_num, kernel_dilation, kernel_dilation_num, input_dilation, input_dilation_num, groups, flip, s);
}

int mlx_conv_transpose1d(
    mlx_array* res,
    const mlx_array input,
    const mlx_array weight,
    int stride,
    int padding,
    int dilation,
    int output_padding,
    int groups,
    const mlx_stream s) {
    return mlx_conv_transpose1d_(res, input, weight, stride, padding, dilation, output_padding, groups, s);
}

int mlx_conv_transpose2d(
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
    const mlx_stream s) {
    return mlx_conv_transpose2d_(res, input, weight, stride_0, stride_1, padding_0, padding_1, dilation_0, dilation_1, output_padding_0, output_padding_1, groups, s);
}

int mlx_conv_transpose3d(
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
    const mlx_stream s) {
    return mlx_conv_transpose3d_(res, input, weight, stride_0, stride_1, stride_2, padding_0, padding_1, padding_2, dilation_0, dilation_1, dilation_2, output_padding_0, output_padding_1, output_padding_2, groups, s);
}

int mlx_copy(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_copy_(res, a, s);
}

int mlx_cos(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_cos_(res, a, s);
}

int mlx_cosh(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_cosh_(res, a, s);
}

int mlx_cummax(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool reverse,
    bool inclusive,
    const mlx_stream s) {
    return mlx_cummax_(res, a, axis, reverse, inclusive, s);
}

int mlx_cummin(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool reverse,
    bool inclusive,
    const mlx_stream s) {
    return mlx_cummin_(res, a, axis, reverse, inclusive, s);
}

int mlx_cumprod(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool reverse,
    bool inclusive,
    const mlx_stream s) {
    return mlx_cumprod_(res, a, axis, reverse, inclusive, s);
}

int mlx_cumsum(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool reverse,
    bool inclusive,
    const mlx_stream s) {
    return mlx_cumsum_(res, a, axis, reverse, inclusive, s);
}

int mlx_degrees(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_degrees_(res, a, s);
}

int mlx_depends(
    mlx_vector_array* res,
    const mlx_vector_array inputs,
    const mlx_vector_array dependencies) {
    return mlx_depends_(res, inputs, dependencies);
}

int mlx_dequantize(
    mlx_array* res,
    const mlx_array w,
    const mlx_array scales,
    const mlx_array biases /* may be null */,
    mlx_optional_int group_size,
    mlx_optional_int bits,
    const char* mode,
    mlx_optional_dtype dtype,
    const mlx_stream s) {
    return mlx_dequantize_(res, w, scales, biases, group_size, bits, mode, dtype, s);
}

int mlx_diag(mlx_array* res, const mlx_array a, int k, const mlx_stream s) {
    return mlx_diag_(res, a, k, s);
}

int mlx_diagonal(
    mlx_array* res,
    const mlx_array a,
    int offset,
    int axis1,
    int axis2,
    const mlx_stream s) {
    return mlx_diagonal_(res, a, offset, axis1, axis2, s);
}

int mlx_divide(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_divide_(res, a, b, s);
}

int mlx_divmod(
    mlx_vector_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_divmod_(res, a, b, s);
}

int mlx_einsum(
    mlx_array* res,
    const char* subscripts,
    const mlx_vector_array operands,
    const mlx_stream s) {
    return mlx_einsum_(res, subscripts, operands, s);
}

int mlx_equal(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_equal_(res, a, b, s);
}

int mlx_erf(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_erf_(res, a, s);
}

int mlx_erfinv(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_erfinv_(res, a, s);
}

int mlx_exp(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_exp_(res, a, s);
}

int mlx_expand_dims_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
    return mlx_expand_dims_axes_(res, a, axes, axes_num, s);
}

int mlx_expand_dims(
    mlx_array* res,
    const mlx_array a,
    int axis,
    const mlx_stream s) {
    return mlx_expand_dims_(res, a, axis, s);
}

int mlx_expm1(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_expm1_(res, a, s);
}

int mlx_eye(
    mlx_array* res,
    int n,
    int m,
    int k,
    mlx_dtype dtype,
    const mlx_stream s) {
    return mlx_eye_(res, n, m, k, dtype, s);
}

int mlx_flatten(
    mlx_array* res,
    const mlx_array a,
    int start_axis,
    int end_axis,
    const mlx_stream s) {
    return mlx_flatten_(res, a, start_axis, end_axis, s);
}

int mlx_floor(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_floor_(res, a, s);
}

int mlx_floor_divide(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_floor_divide_(res, a, b, s);
}

int mlx_from_fp8(
    mlx_array* res,
    const mlx_array x,
    mlx_dtype dtype,
    const mlx_stream s) {
    return mlx_from_fp8_(res, x, dtype, s);
}

int mlx_full(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    const mlx_array vals,
    mlx_dtype dtype,
    const mlx_stream s) {
    return mlx_full_(res, shape, shape_num, vals, dtype, s);
}

int mlx_full_like(
    mlx_array* res,
    const mlx_array a,
    const mlx_array vals,
    mlx_dtype dtype,
    const mlx_stream s) {
    return mlx_full_like_(res, a, vals, dtype, s);
}

int mlx_gather(
    mlx_array* res,
    const mlx_array a,
    const mlx_vector_array indices,
    const int* axes,
    size_t axes_num,
    const int* slice_sizes,
    size_t slice_sizes_num,
    const mlx_stream s) {
    return mlx_gather_(res, a, indices, axes, axes_num, slice_sizes, slice_sizes_num, s);
}

int mlx_gather_mm(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_array lhs_indices /* may be null */,
    const mlx_array rhs_indices /* may be null */,
    bool sorted_indices,
    const mlx_stream s) {
    return mlx_gather_mm_(res, a, b, lhs_indices, rhs_indices, sorted_indices, s);
}

int mlx_gather_qmm(
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
    const mlx_stream s) {
    return mlx_gather_qmm_(res, x, w, scales, biases, lhs_indices, rhs_indices, transpose, group_size, bits, mode, sorted_indices, s);
}

int mlx_greater(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_greater_(res, a, b, s);
}

int mlx_greater_equal(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_greater_equal_(res, a, b, s);
}

int mlx_hadamard_transform(
    mlx_array* res,
    const mlx_array a,
    mlx_optional_float scale,
    const mlx_stream s) {
    return mlx_hadamard_transform_(res, a, scale, s);
}

int mlx_identity(mlx_array* res, int n, mlx_dtype dtype, const mlx_stream s) {
    return mlx_identity_(res, n, dtype, s);
}

int mlx_imag(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_imag_(res, a, s);
}

int mlx_inner(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_inner_(res, a, b, s);
}

int mlx_isclose(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    double rtol,
    double atol,
    bool equal_nan,
    const mlx_stream s) {
    return mlx_isclose_(res, a, b, rtol, atol, equal_nan, s);
}

int mlx_isfinite(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_isfinite_(res, a, s);
}

int mlx_isinf(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_isinf_(res, a, s);
}

int mlx_isnan(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_isnan_(res, a, s);
}

int mlx_isneginf(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_isneginf_(res, a, s);
}

int mlx_isposinf(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_isposinf_(res, a, s);
}

int mlx_kron(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_kron_(res, a, b, s);
}

int mlx_left_shift(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_left_shift_(res, a, b, s);
}

int mlx_less(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_less_(res, a, b, s);
}

int mlx_less_equal(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_less_equal_(res, a, b, s);
}

int mlx_linspace(
    mlx_array* res,
    double start,
    double stop,
    int num,
    mlx_dtype dtype,
    const mlx_stream s) {
    return mlx_linspace_(res, start, stop, num, dtype, s);
}

int mlx_log(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_log_(res, a, s);
}

int mlx_log10(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_log10_(res, a, s);
}

int mlx_log1p(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_log1p_(res, a, s);
}

int mlx_log2(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_log2_(res, a, s);
}

int mlx_logaddexp(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_logaddexp_(res, a, b, s);
}

int mlx_logcumsumexp(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool reverse,
    bool inclusive,
    const mlx_stream s) {
    return mlx_logcumsumexp_(res, a, axis, reverse, inclusive, s);
}

int mlx_logical_and(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_logical_and_(res, a, b, s);
}

int mlx_logical_not(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_logical_not_(res, a, s);
}

int mlx_logical_or(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_logical_or_(res, a, b, s);
}

int mlx_logsumexp_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s) {
    return mlx_logsumexp_axes_(res, a, axes, axes_num, keepdims, s);
}

int mlx_logsumexp_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s) {
    return mlx_logsumexp_axis_(res, a, axis, keepdims, s);
}

int mlx_logsumexp(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s) {
    return mlx_logsumexp_(res, a, keepdims, s);
}

int mlx_masked_scatter(
    mlx_array* res,
    const mlx_array a,
    const mlx_array mask,
    const mlx_array src,
    const mlx_stream s) {
    return mlx_masked_scatter_(res, a, mask, src, s);
}

int mlx_matmul(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_matmul_(res, a, b, s);
}

int mlx_max_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s) {
    return mlx_max_axes_(res, a, axes, axes_num, keepdims, s);
}

int mlx_max_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s) {
    return mlx_max_axis_(res, a, axis, keepdims, s);
}

int mlx_max(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s) {
    return mlx_max_(res, a, keepdims, s);
}

int mlx_maximum(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_maximum_(res, a, b, s);
}

int mlx_mean_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s) {
    return mlx_mean_axes_(res, a, axes, axes_num, keepdims, s);
}

int mlx_mean_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s) {
    return mlx_mean_axis_(res, a, axis, keepdims, s);
}

int mlx_mean(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s) {
    return mlx_mean_(res, a, keepdims, s);
}

int mlx_median(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s) {
    return mlx_median_(res, a, axes, axes_num, keepdims, s);
}

int mlx_meshgrid(
    mlx_vector_array* res,
    const mlx_vector_array arrays,
    bool sparse,
    const char* indexing,
    const mlx_stream s) {
    return mlx_meshgrid_(res, arrays, sparse, indexing, s);
}

int mlx_min_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s) {
    return mlx_min_axes_(res, a, axes, axes_num, keepdims, s);
}

int mlx_min_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s) {
    return mlx_min_axis_(res, a, axis, keepdims, s);
}

int mlx_min(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s) {
    return mlx_min_(res, a, keepdims, s);
}

int mlx_minimum(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_minimum_(res, a, b, s);
}

int mlx_moveaxis(
    mlx_array* res,
    const mlx_array a,
    int source,
    int destination,
    const mlx_stream s) {
    return mlx_moveaxis_(res, a, source, destination, s);
}

int mlx_multiply(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_multiply_(res, a, b, s);
}

int mlx_nan_to_num(
    mlx_array* res,
    const mlx_array a,
    float nan,
    mlx_optional_float posinf,
    mlx_optional_float neginf,
    const mlx_stream s) {
    return mlx_nan_to_num_(res, a, nan, posinf, neginf, s);
}

int mlx_negative(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_negative_(res, a, s);
}

int mlx_not_equal(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_not_equal_(res, a, b, s);
}

int mlx_number_of_elements(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool inverted,
    mlx_dtype dtype,
    const mlx_stream s) {
    return mlx_number_of_elements_(res, a, axes, axes_num, inverted, dtype, s);
}

int mlx_ones(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_stream s) {
    return mlx_ones_(res, shape, shape_num, dtype, s);
}

int mlx_ones_like(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_ones_like_(res, a, s);
}

int mlx_outer(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_outer_(res, a, b, s);
}

int mlx_pad(
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
    const mlx_stream s) {
    return mlx_pad_(res, a, axes, axes_num, low_pad_size, low_pad_size_num, high_pad_size, high_pad_size_num, pad_value, mode, s);
}

int mlx_pad_symmetric(
    mlx_array* res,
    const mlx_array a,
    int pad_width,
    const mlx_array pad_value,
    const char* mode,
    const mlx_stream s) {
    return mlx_pad_symmetric_(res, a, pad_width, pad_value, mode, s);
}

int mlx_partition_axis(
    mlx_array* res,
    const mlx_array a,
    int kth,
    int axis,
    const mlx_stream s) {
    return mlx_partition_axis_(res, a, kth, axis, s);
}

int mlx_partition(
    mlx_array* res,
    const mlx_array a,
    int kth,
    const mlx_stream s) {
    return mlx_partition_(res, a, kth, s);
}

int mlx_power(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_power_(res, a, b, s);
}

int mlx_prod_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s) {
    return mlx_prod_axes_(res, a, axes, axes_num, keepdims, s);
}

int mlx_prod_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s) {
    return mlx_prod_axis_(res, a, axis, keepdims, s);
}

int mlx_prod(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s) {
    return mlx_prod_(res, a, keepdims, s);
}

int mlx_put_along_axis(
    mlx_array* res,
    const mlx_array a,
    const mlx_array indices,
    const mlx_array values,
    int axis,
    const mlx_stream s) {
    return mlx_put_along_axis_(res, a, indices, values, axis, s);
}

int mlx_quantize(
    mlx_vector_array* res,
    const mlx_array w,
    mlx_optional_int group_size,
    mlx_optional_int bits,
    const char* mode,
    const mlx_stream s) {
    return mlx_quantize_(res, w, group_size, bits, mode, s);
}

int mlx_quantized_matmul(
    mlx_array* res,
    const mlx_array x,
    const mlx_array w,
    const mlx_array scales,
    const mlx_array biases /* may be null */,
    bool transpose,
    mlx_optional_int group_size,
    mlx_optional_int bits,
    const char* mode,
    const mlx_stream s) {
    return mlx_quantized_matmul_(res, x, w, scales, biases, transpose, group_size, bits, mode, s);
}

int mlx_radians(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_radians_(res, a, s);
}

int mlx_real(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_real_(res, a, s);
}

int mlx_reciprocal(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_reciprocal_(res, a, s);
}

int mlx_remainder(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_remainder_(res, a, b, s);
}

int mlx_repeat_axis(
    mlx_array* res,
    const mlx_array arr,
    int repeats,
    int axis,
    const mlx_stream s) {
    return mlx_repeat_axis_(res, arr, repeats, axis, s);
}

int mlx_repeat(
    mlx_array* res,
    const mlx_array arr,
    int repeats,
    const mlx_stream s) {
    return mlx_repeat_(res, arr, repeats, s);
}

int mlx_reshape(
    mlx_array* res,
    const mlx_array a,
    const int* shape,
    size_t shape_num,
    const mlx_stream s) {
    return mlx_reshape_(res, a, shape, shape_num, s);
}

int mlx_right_shift(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_right_shift_(res, a, b, s);
}

int mlx_roll_axis(
    mlx_array* res,
    const mlx_array a,
    const int* shift,
    size_t shift_num,
    int axis,
    const mlx_stream s) {
    return mlx_roll_axis_(res, a, shift, shift_num, axis, s);
}

int mlx_roll_axes(
    mlx_array* res,
    const mlx_array a,
    const int* shift,
    size_t shift_num,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
    return mlx_roll_axes_(res, a, shift, shift_num, axes, axes_num, s);
}

int mlx_roll(
    mlx_array* res,
    const mlx_array a,
    const int* shift,
    size_t shift_num,
    const mlx_stream s) {
    return mlx_roll_(res, a, shift, shift_num, s);
}

int mlx_round(
    mlx_array* res,
    const mlx_array a,
    int decimals,
    const mlx_stream s) {
    return mlx_round_(res, a, decimals, s);
}

int mlx_rsqrt(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_rsqrt_(res, a, s);
}

int mlx_scatter(
    mlx_array* res,
    const mlx_array a,
    const mlx_vector_array indices,
    const mlx_array updates,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
    return mlx_scatter_(res, a, indices, updates, axes, axes_num, s);
}

int mlx_scatter_add(
    mlx_array* res,
    const mlx_array a,
    const mlx_vector_array indices,
    const mlx_array updates,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
    return mlx_scatter_add_(res, a, indices, updates, axes, axes_num, s);
}

int mlx_scatter_add_axis(
    mlx_array* res,
    const mlx_array a,
    const mlx_array indices,
    const mlx_array values,
    int axis,
    const mlx_stream s) {
    return mlx_scatter_add_axis_(res, a, indices, values, axis, s);
}

int mlx_scatter_max(
    mlx_array* res,
    const mlx_array a,
    const mlx_vector_array indices,
    const mlx_array updates,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
    return mlx_scatter_max_(res, a, indices, updates, axes, axes_num, s);
}

int mlx_scatter_min(
    mlx_array* res,
    const mlx_array a,
    const mlx_vector_array indices,
    const mlx_array updates,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
    return mlx_scatter_min_(res, a, indices, updates, axes, axes_num, s);
}

int mlx_scatter_prod(
    mlx_array* res,
    const mlx_array a,
    const mlx_vector_array indices,
    const mlx_array updates,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
    return mlx_scatter_prod_(res, a, indices, updates, axes, axes_num, s);
}

int mlx_segmented_mm(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_array segments,
    const mlx_stream s) {
    return mlx_segmented_mm_(res, a, b, segments, s);
}

int mlx_sigmoid(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_sigmoid_(res, a, s);
}

int mlx_sign(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_sign_(res, a, s);
}

int mlx_sin(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_sin_(res, a, s);
}

int mlx_sinh(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_sinh_(res, a, s);
}

int mlx_slice(
    mlx_array* res,
    const mlx_array a,
    const int* start,
    size_t start_num,
    const int* stop,
    size_t stop_num,
    const int* strides,
    size_t strides_num,
    const mlx_stream s) {
    return mlx_slice_(res, a, start, start_num, stop, stop_num, strides, strides_num, s);
}

int mlx_slice_dynamic(
    mlx_array* res,
    const mlx_array a,
    const mlx_array start,
    const int* axes,
    size_t axes_num,
    const int* slice_size,
    size_t slice_size_num,
    const mlx_stream s) {
    return mlx_slice_dynamic_(res, a, start, axes, axes_num, slice_size, slice_size_num, s);
}

int mlx_slice_update(
    mlx_array* res,
    const mlx_array src,
    const mlx_array update,
    const int* start,
    size_t start_num,
    const int* stop,
    size_t stop_num,
    const int* strides,
    size_t strides_num,
    const mlx_stream s) {
    return mlx_slice_update_(res, src, update, start, start_num, stop, stop_num, strides, strides_num, s);
}

int mlx_slice_update_dynamic(
    mlx_array* res,
    const mlx_array src,
    const mlx_array update,
    const mlx_array start,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
    return mlx_slice_update_dynamic_(res, src, update, start, axes, axes_num, s);
}

int mlx_softmax_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool precise,
    const mlx_stream s) {
    return mlx_softmax_axes_(res, a, axes, axes_num, precise, s);
}

int mlx_softmax_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool precise,
    const mlx_stream s) {
    return mlx_softmax_axis_(res, a, axis, precise, s);
}

int mlx_softmax(
    mlx_array* res,
    const mlx_array a,
    bool precise,
    const mlx_stream s) {
    return mlx_softmax_(res, a, precise, s);
}

int mlx_sort_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    const mlx_stream s) {
    return mlx_sort_axis_(res, a, axis, s);
}

int mlx_sort(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_sort_(res, a, s);
}

int mlx_split(
    mlx_vector_array* res,
    const mlx_array a,
    int num_splits,
    int axis,
    const mlx_stream s) {
    return mlx_split_(res, a, num_splits, axis, s);
}

int mlx_split_sections(
    mlx_vector_array* res,
    const mlx_array a,
    const int* indices,
    size_t indices_num,
    int axis,
    const mlx_stream s) {
    return mlx_split_sections_(res, a, indices, indices_num, axis, s);
}

int mlx_sqrt(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_sqrt_(res, a, s);
}

int mlx_square(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_square_(res, a, s);
}

int mlx_squeeze_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
    return mlx_squeeze_axes_(res, a, axes, axes_num, s);
}

int mlx_squeeze_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    const mlx_stream s) {
    return mlx_squeeze_axis_(res, a, axis, s);
}

int mlx_squeeze(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_squeeze_(res, a, s);
}

int mlx_stack_axis(
    mlx_array* res,
    const mlx_vector_array arrays,
    int axis,
    const mlx_stream s) {
    return mlx_stack_axis_(res, arrays, axis, s);
}

int mlx_stack(
    mlx_array* res,
    const mlx_vector_array arrays,
    const mlx_stream s) {
    return mlx_stack_(res, arrays, s);
}

int mlx_std_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    int ddof,
    const mlx_stream s) {
    return mlx_std_axes_(res, a, axes, axes_num, keepdims, ddof, s);
}

int mlx_std_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    int ddof,
    const mlx_stream s) {
    return mlx_std_axis_(res, a, axis, keepdims, ddof, s);
}

int mlx_std(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    int ddof,
    const mlx_stream s) {
    return mlx_std_(res, a, keepdims, ddof, s);
}

int mlx_stop_gradient(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_stop_gradient_(res, a, s);
}

int mlx_subtract(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const mlx_stream s) {
    return mlx_subtract_(res, a, b, s);
}

int mlx_sum_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const mlx_stream s) {
    return mlx_sum_axes_(res, a, axes, axes_num, keepdims, s);
}

int mlx_sum_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    const mlx_stream s) {
    return mlx_sum_axis_(res, a, axis, keepdims, s);
}

int mlx_sum(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    const mlx_stream s) {
    return mlx_sum_(res, a, keepdims, s);
}

int mlx_swapaxes(
    mlx_array* res,
    const mlx_array a,
    int axis1,
    int axis2,
    const mlx_stream s) {
    return mlx_swapaxes_(res, a, axis1, axis2, s);
}

int mlx_take_axis(
    mlx_array* res,
    const mlx_array a,
    const mlx_array indices,
    int axis,
    const mlx_stream s) {
    return mlx_take_axis_(res, a, indices, axis, s);
}

int mlx_take(
    mlx_array* res,
    const mlx_array a,
    const mlx_array indices,
    const mlx_stream s) {
    return mlx_take_(res, a, indices, s);
}

int mlx_take_along_axis(
    mlx_array* res,
    const mlx_array a,
    const mlx_array indices,
    int axis,
    const mlx_stream s) {
    return mlx_take_along_axis_(res, a, indices, axis, s);
}

int mlx_tan(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_tan_(res, a, s);
}

int mlx_tanh(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_tanh_(res, a, s);
}

int mlx_tensordot(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    const int* axes_a,
    size_t axes_a_num,
    const int* axes_b,
    size_t axes_b_num,
    const mlx_stream s) {
    return mlx_tensordot_(res, a, b, axes_a, axes_a_num, axes_b, axes_b_num, s);
}

int mlx_tensordot_axis(
    mlx_array* res,
    const mlx_array a,
    const mlx_array b,
    int axis,
    const mlx_stream s) {
    return mlx_tensordot_axis_(res, a, b, axis, s);
}

int mlx_tile(
    mlx_array* res,
    const mlx_array arr,
    const int* reps,
    size_t reps_num,
    const mlx_stream s) {
    return mlx_tile_(res, arr, reps, reps_num, s);
}

int mlx_to_fp8(mlx_array* res, const mlx_array x, const mlx_stream s) {
    return mlx_to_fp8_(res, x, s);
}

int mlx_topk_axis(
    mlx_array* res,
    const mlx_array a,
    int k,
    int axis,
    const mlx_stream s) {
    return mlx_topk_axis_(res, a, k, axis, s);
}

int mlx_topk(mlx_array* res, const mlx_array a, int k, const mlx_stream s) {
    return mlx_topk_(res, a, k, s);
}

int mlx_trace(
    mlx_array* res,
    const mlx_array a,
    int offset,
    int axis1,
    int axis2,
    mlx_dtype dtype,
    const mlx_stream s) {
    return mlx_trace_(res, a, offset, axis1, axis2, dtype, s);
}

int mlx_transpose_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    const mlx_stream s) {
    return mlx_transpose_axes_(res, a, axes, axes_num, s);
}

int mlx_transpose(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_transpose_(res, a, s);
}

int mlx_tri(
    mlx_array* res,
    int n,
    int m,
    int k,
    mlx_dtype type,
    const mlx_stream s) {
    return mlx_tri_(res, n, m, k, type, s);
}

int mlx_tril(mlx_array* res, const mlx_array x, int k, const mlx_stream s) {
    return mlx_tril_(res, x, k, s);
}

int mlx_triu(mlx_array* res, const mlx_array x, int k, const mlx_stream s) {
    return mlx_triu_(res, x, k, s);
}

int mlx_unflatten(
    mlx_array* res,
    const mlx_array a,
    int axis,
    const int* shape,
    size_t shape_num,
    const mlx_stream s) {
    return mlx_unflatten_(res, a, axis, shape, shape_num, s);
}

int mlx_var_axes(
    mlx_array* res,
    const mlx_array a,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    int ddof,
    const mlx_stream s) {
    return mlx_var_axes_(res, a, axes, axes_num, keepdims, ddof, s);
}

int mlx_var_axis(
    mlx_array* res,
    const mlx_array a,
    int axis,
    bool keepdims,
    int ddof,
    const mlx_stream s) {
    return mlx_var_axis_(res, a, axis, keepdims, ddof, s);
}

int mlx_var(
    mlx_array* res,
    const mlx_array a,
    bool keepdims,
    int ddof,
    const mlx_stream s) {
    return mlx_var_(res, a, keepdims, ddof, s);
}

int mlx_view(
    mlx_array* res,
    const mlx_array a,
    mlx_dtype dtype,
    const mlx_stream s) {
    return mlx_view_(res, a, dtype, s);
}

int mlx_where(
    mlx_array* res,
    const mlx_array condition,
    const mlx_array x,
    const mlx_array y,
    const mlx_stream s) {
    return mlx_where_(res, condition, x, y, s);
}

int mlx_zeros(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_stream s) {
    return mlx_zeros_(res, shape, shape_num, dtype, s);
}

int mlx_zeros_like(mlx_array* res, const mlx_array a, const mlx_stream s) {
    return mlx_zeros_like_(res, a, s);
}

int mlx_random_bernoulli(
    mlx_array* res,
    const mlx_array p,
    const int* shape,
    size_t shape_num,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
    return mlx_random_bernoulli_(res, p, shape, shape_num, key, s);
}

int mlx_random_bits(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    int width,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
    return mlx_random_bits_(res, shape, shape_num, width, key, s);
}

int mlx_random_categorical_shape(
    mlx_array* res,
    const mlx_array logits,
    int axis,
    const int* shape,
    size_t shape_num,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
    return mlx_random_categorical_shape_(res, logits, axis, shape, shape_num, key, s);
}

int mlx_random_categorical_num_samples(
    mlx_array* res,
    const mlx_array logits_,
    int axis,
    int num_samples,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
    return mlx_random_categorical_num_samples_(res, logits_, axis, num_samples, key, s);
}

int mlx_random_categorical(
    mlx_array* res,
    const mlx_array logits,
    int axis,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
    return mlx_random_categorical_(res, logits, axis, key, s);
}

int mlx_random_gumbel(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
    return mlx_random_gumbel_(res, shape, shape_num, dtype, key, s);
}

int mlx_random_key(mlx_array* res, uint64_t seed) {
    return mlx_random_key_(res, seed);
}

int mlx_random_laplace(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    float loc,
    float scale,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
    return mlx_random_laplace_(res, shape, shape_num, dtype, loc, scale, key, s);
}

int mlx_random_multivariate_normal(
    mlx_array* res,
    const mlx_array mean,
    const mlx_array cov,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
    return mlx_random_multivariate_normal_(res, mean, cov, shape, shape_num, dtype, key, s);
}

int mlx_random_normal_broadcast(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array loc /* may be null */,
    const mlx_array scale /* may be null */,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
    return mlx_random_normal_broadcast_(res, shape, shape_num, dtype, loc, scale, key, s);
}

int mlx_random_normal(
    mlx_array* res,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    float loc,
    float scale,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
    return mlx_random_normal_(res, shape, shape_num, dtype, loc, scale, key, s);
}

int mlx_random_permutation(
    mlx_array* res,
    const mlx_array x,
    int axis,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
    return mlx_random_permutation_(res, x, axis, key, s);
}

int mlx_random_permutation_arange(
    mlx_array* res,
    int x,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
    return mlx_random_permutation_arange_(res, x, key, s);
}

int mlx_random_randint(
    mlx_array* res,
    const mlx_array low,
    const mlx_array high,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
    return mlx_random_randint_(res, low, high, shape, shape_num, dtype, key, s);
}

int mlx_random_seed(uint64_t seed) {
    return mlx_random_seed_(seed);
}

int mlx_random_split_num(
    mlx_array* res,
    const mlx_array key,
    int num,
    const mlx_stream s) {
    return mlx_random_split_num_(res, key, num, s);
}

int mlx_random_split(
    mlx_array* res_0,
    mlx_array* res_1,
    const mlx_array key,
    const mlx_stream s) {
    return mlx_random_split_(res_0, res_1, key, s);
}

int mlx_random_truncated_normal(
    mlx_array* res,
    const mlx_array lower,
    const mlx_array upper,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
    return mlx_random_truncated_normal_(res, lower, upper, shape, shape_num, dtype, key, s);
}

int mlx_random_uniform(
    mlx_array* res,
    const mlx_array low,
    const mlx_array high,
    const int* shape,
    size_t shape_num,
    mlx_dtype dtype,
    const mlx_array key /* may be null */,
    const mlx_stream s) {
    return mlx_random_uniform_(res, low, high, shape, shape_num, dtype, key, s);
}

mlx_stream mlx_stream_new(void) {
    return mlx_stream_new_();
}

mlx_stream mlx_stream_new_device(mlx_device dev) {
    return mlx_stream_new_device_(dev);
}

int mlx_stream_set(mlx_stream* stream, const mlx_stream src) {
    return mlx_stream_set_(stream, src);
}

int mlx_stream_free(mlx_stream stream) {
    return mlx_stream_free_(stream);
}

int mlx_stream_tostring(mlx_string* str, mlx_stream stream) {
    return mlx_stream_tostring_(str, stream);
}

bool mlx_stream_equal(mlx_stream lhs, mlx_stream rhs) {
    return mlx_stream_equal_(lhs, rhs);
}

int mlx_stream_get_device(mlx_device* dev, mlx_stream stream) {
    return mlx_stream_get_device_(dev, stream);
}

int mlx_stream_get_index(int* index, mlx_stream stream) {
    return mlx_stream_get_index_(index, stream);
}

int mlx_synchronize(mlx_stream stream) {
    return mlx_synchronize_(stream);
}

int mlx_get_default_stream(mlx_stream* stream, mlx_device dev) {
    return mlx_get_default_stream_(stream, dev);
}

int mlx_set_default_stream(mlx_stream stream) {
    return mlx_set_default_stream_(stream);
}

mlx_stream mlx_default_cpu_stream_new(void) {
    return mlx_default_cpu_stream_new_();
}

mlx_stream mlx_default_gpu_stream_new(void) {
    return mlx_default_gpu_stream_new_();
}

mlx_string mlx_string_new(void) {
    return mlx_string_new_();
}

mlx_string mlx_string_new_data(const char* str) {
    return mlx_string_new_data_(str);
}

int mlx_string_set(mlx_string* str, const mlx_string src) {
    return mlx_string_set_(str, src);
}

const char * mlx_string_data(mlx_string str) {
    return mlx_string_data_(str);
}

int mlx_string_free(mlx_string str) {
    return mlx_string_free_(str);
}

int mlx_detail_vmap_replace(
    mlx_vector_array* res,
    const mlx_vector_array inputs,
    const mlx_vector_array s_inputs,
    const mlx_vector_array s_outputs,
    const int* in_axes,
    size_t in_axes_num,
    const int* out_axes,
    size_t out_axes_num) {
    return mlx_detail_vmap_replace_(res, inputs, s_inputs, s_outputs, in_axes, in_axes_num, out_axes, out_axes_num);
}

int mlx_detail_vmap_trace(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    const mlx_closure fun,
    const mlx_vector_array inputs,
    const int* in_axes,
    size_t in_axes_num) {
    return mlx_detail_vmap_trace_(res_0, res_1, fun, inputs, in_axes, in_axes_num);
}

int mlx_async_eval(const mlx_vector_array outputs) {
    return mlx_async_eval_(outputs);
}

int mlx_checkpoint(mlx_closure* res, const mlx_closure fun) {
    return mlx_checkpoint_(res, fun);
}

int mlx_custom_function(
    mlx_closure* res,
    const mlx_closure fun,
    const mlx_closure_custom fun_vjp /* may be null */,
    const mlx_closure_custom_jvp fun_jvp /* may be null */,
    const mlx_closure_custom_vmap fun_vmap /* may be null */) {
    return mlx_custom_function_(res, fun, fun_vjp, fun_jvp, fun_vmap);
}

int mlx_custom_vjp(
    mlx_closure* res,
    const mlx_closure fun,
    const mlx_closure_custom fun_vjp) {
    return mlx_custom_vjp_(res, fun, fun_vjp);
}

int mlx_eval(const mlx_vector_array outputs) {
    return mlx_eval_(outputs);
}

int mlx_jvp(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    const mlx_closure fun,
    const mlx_vector_array primals,
    const mlx_vector_array tangents) {
    return mlx_jvp_(res_0, res_1, fun, primals, tangents);
}

int mlx_value_and_grad(
    mlx_closure_value_and_grad* res,
    const mlx_closure fun,
    const int* argnums,
    size_t argnums_num) {
    return mlx_value_and_grad_(res, fun, argnums, argnums_num);
}

int mlx_vjp(
    mlx_vector_array* res_0,
    mlx_vector_array* res_1,
    const mlx_closure fun,
    const mlx_vector_array primals,
    const mlx_vector_array cotangents) {
    return mlx_vjp_(res_0, res_1, fun, primals, cotangents);
}

mlx_vector_array mlx_vector_array_new(void) {
    return mlx_vector_array_new_();
}

int mlx_vector_array_set(mlx_vector_array* vec, const mlx_vector_array src) {
    return mlx_vector_array_set_(vec, src);
}

int mlx_vector_array_free(mlx_vector_array vec) {
    return mlx_vector_array_free_(vec);
}

mlx_vector_array mlx_vector_array_new_data(const mlx_array* data, size_t size) {
    return mlx_vector_array_new_data_(data, size);
}

mlx_vector_array mlx_vector_array_new_value(const mlx_array val) {
    return mlx_vector_array_new_value_(val);
}

int mlx_vector_array_set_data(
    mlx_vector_array* vec,
    const mlx_array* data,
    size_t size) {
    return mlx_vector_array_set_data_(vec, data, size);
}

int mlx_vector_array_set_value(mlx_vector_array* vec, const mlx_array val) {
    return mlx_vector_array_set_value_(vec, val);
}

int mlx_vector_array_append_data(
    mlx_vector_array vec,
    const mlx_array* data,
    size_t size) {
    return mlx_vector_array_append_data_(vec, data, size);
}

int mlx_vector_array_append_value(mlx_vector_array vec, const mlx_array val) {
    return mlx_vector_array_append_value_(vec, val);
}

size_t mlx_vector_array_size(mlx_vector_array vec) {
    return mlx_vector_array_size_(vec);
}

int mlx_vector_array_get(
    mlx_array* res,
    const mlx_vector_array vec,
    size_t idx) {
    return mlx_vector_array_get_(res, vec, idx);
}

mlx_vector_vector_array mlx_vector_vector_array_new(void) {
    return mlx_vector_vector_array_new_();
}

int mlx_vector_vector_array_set(
    mlx_vector_vector_array* vec,
    const mlx_vector_vector_array src) {
    return mlx_vector_vector_array_set_(vec, src);
}

int mlx_vector_vector_array_free(mlx_vector_vector_array vec) {
    return mlx_vector_vector_array_free_(vec);
}

mlx_vector_vector_array mlx_vector_vector_array_new_data(
    const mlx_vector_array* data,
    size_t size) {
    return mlx_vector_vector_array_new_data_(data, size);
}

mlx_vector_vector_array mlx_vector_vector_array_new_value(
    const mlx_vector_array val) {
    return mlx_vector_vector_array_new_value_(val);
}

int mlx_vector_vector_array_set_data(
    mlx_vector_vector_array* vec,
    const mlx_vector_array* data,
    size_t size) {
    return mlx_vector_vector_array_set_data_(vec, data, size);
}

int mlx_vector_vector_array_set_value(
    mlx_vector_vector_array* vec,
    const mlx_vector_array val) {
    return mlx_vector_vector_array_set_value_(vec, val);
}

int mlx_vector_vector_array_append_data(
    mlx_vector_vector_array vec,
    const mlx_vector_array* data,
    size_t size) {
    return mlx_vector_vector_array_append_data_(vec, data, size);
}

int mlx_vector_vector_array_append_value(
    mlx_vector_vector_array vec,
    const mlx_vector_array val) {
    return mlx_vector_vector_array_append_value_(vec, val);
}

size_t mlx_vector_vector_array_size(mlx_vector_vector_array vec) {
    return mlx_vector_vector_array_size_(vec);
}

int mlx_vector_vector_array_get(
    mlx_vector_array* res,
    const mlx_vector_vector_array vec,
    size_t idx) {
    return mlx_vector_vector_array_get_(res, vec, idx);
}

mlx_vector_int mlx_vector_int_new(void) {
    return mlx_vector_int_new_();
}

int mlx_vector_int_set(mlx_vector_int* vec, const mlx_vector_int src) {
    return mlx_vector_int_set_(vec, src);
}

int mlx_vector_int_free(mlx_vector_int vec) {
    return mlx_vector_int_free_(vec);
}

mlx_vector_int mlx_vector_int_new_data(int* data, size_t size) {
    return mlx_vector_int_new_data_(data, size);
}

mlx_vector_int mlx_vector_int_new_value(int val) {
    return mlx_vector_int_new_value_(val);
}

int mlx_vector_int_set_data(mlx_vector_int* vec, int* data, size_t size) {
    return mlx_vector_int_set_data_(vec, data, size);
}

int mlx_vector_int_set_value(mlx_vector_int* vec, int val) {
    return mlx_vector_int_set_value_(vec, val);
}

int mlx_vector_int_append_data(mlx_vector_int vec, int* data, size_t size) {
    return mlx_vector_int_append_data_(vec, data, size);
}

int mlx_vector_int_append_value(mlx_vector_int vec, int val) {
    return mlx_vector_int_append_value_(vec, val);
}

size_t mlx_vector_int_size(mlx_vector_int vec) {
    return mlx_vector_int_size_(vec);
}

int mlx_vector_int_get(int* res, const mlx_vector_int vec, size_t idx) {
    return mlx_vector_int_get_(res, vec, idx);
}

mlx_vector_string mlx_vector_string_new(void) {
    return mlx_vector_string_new_();
}

int mlx_vector_string_set(mlx_vector_string* vec, const mlx_vector_string src) {
    return mlx_vector_string_set_(vec, src);
}

int mlx_vector_string_free(mlx_vector_string vec) {
    return mlx_vector_string_free_(vec);
}

mlx_vector_string mlx_vector_string_new_data(const char** data, size_t size) {
    return mlx_vector_string_new_data_(data, size);
}

mlx_vector_string mlx_vector_string_new_value(const char* val) {
    return mlx_vector_string_new_value_(val);
}

int mlx_vector_string_set_data(
    mlx_vector_string* vec,
    const char** data,
    size_t size) {
    return mlx_vector_string_set_data_(vec, data, size);
}

int mlx_vector_string_set_value(mlx_vector_string* vec, const char* val) {
    return mlx_vector_string_set_value_(vec, val);
}

int mlx_vector_string_append_data(
    mlx_vector_string vec,
    const char** data,
    size_t size) {
    return mlx_vector_string_append_data_(vec, data, size);
}

int mlx_vector_string_append_value(mlx_vector_string vec, const char* val) {
    return mlx_vector_string_append_value_(vec, val);
}

size_t mlx_vector_string_size(mlx_vector_string vec) {
    return mlx_vector_string_size_(vec);
}

int mlx_vector_string_get(char** res, const mlx_vector_string vec, size_t idx) {
    return mlx_vector_string_get_(res, vec, idx);
}

int mlx_version(mlx_string* str_) {
    return mlx_version_(str_);
}
