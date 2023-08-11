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

#import "ggml-metal.h"

#import "ggml.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#ifdef GGML_METAL_NDEBUG
#define metal_printf(...)
#else
#define metal_printf(...) fprintf(stderr, __VA_ARGS__)
#endif

#define UNUSED(x) (void)(x)

struct ggml_metal_buffer {
    const char * name;

    void   * data;
    size_t   size;

    id<MTLBuffer> metal;
};

struct ggml_metal_context {
    int n_cb;

    float * logits;

    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary>      library;

    int n_buffers;
    struct ggml_metal_buffer buffers[GGML_METAL_MAX_BUFFERS];

    int concur_list[GGML_MAX_NODES];
    int concur_list_len;

    // custom kernels
#define GGML_METAL_DECL_KERNEL(name) \
    id<MTLFunction>             function_##name; \
    id<MTLComputePipelineState> pipeline_##name

    GGML_METAL_DECL_KERNEL(add);
    GGML_METAL_DECL_KERNEL(add_row); // TODO: avoid this extra kernel, instead extend the "add" kernel to support broadcast
    GGML_METAL_DECL_KERNEL(mul);
    GGML_METAL_DECL_KERNEL(mul_row); // TODO: avoid this extra kernel, instead extend the "mul" kernel to support broadcast
    GGML_METAL_DECL_KERNEL(scale);
    GGML_METAL_DECL_KERNEL(silu);
    GGML_METAL_DECL_KERNEL(relu);
    GGML_METAL_DECL_KERNEL(gelu);
    GGML_METAL_DECL_KERNEL(soft_max);
    GGML_METAL_DECL_KERNEL(diag_mask_inf);
    GGML_METAL_DECL_KERNEL(get_rows_f16);
    GGML_METAL_DECL_KERNEL(get_rows_q4_0);
    GGML_METAL_DECL_KERNEL(get_rows_q4_1);
    GGML_METAL_DECL_KERNEL(get_rows_q2_K);
    GGML_METAL_DECL_KERNEL(get_rows_q3_K);
    GGML_METAL_DECL_KERNEL(get_rows_q4_K);
    GGML_METAL_DECL_KERNEL(get_rows_q5_K);
    GGML_METAL_DECL_KERNEL(get_rows_q6_K);
    GGML_METAL_DECL_KERNEL(rms_norm);
    GGML_METAL_DECL_KERNEL(norm);
    GGML_METAL_DECL_KERNEL(mul_mat_f16_f32);
    GGML_METAL_DECL_KERNEL(mul_mat_q4_0_f32);
    GGML_METAL_DECL_KERNEL(mul_mat_q4_1_f32);
    GGML_METAL_DECL_KERNEL(mul_mat_q2_K_f32);
    GGML_METAL_DECL_KERNEL(mul_mat_q3_K_f32);
    GGML_METAL_DECL_KERNEL(mul_mat_q4_K_f32);
    GGML_METAL_DECL_KERNEL(mul_mat_q5_K_f32);
    GGML_METAL_DECL_KERNEL(mul_mat_q6_K_f32);
    GGML_METAL_DECL_KERNEL(rope);
    GGML_METAL_DECL_KERNEL(alibi_f32);
    GGML_METAL_DECL_KERNEL(cpy_f32_f16);
    GGML_METAL_DECL_KERNEL(cpy_f32_f32);
    GGML_METAL_DECL_KERNEL(cpy_f16_f16);

#undef GGML_METAL_DECL_KERNEL
};

// MSL code
// TODO: move the contents here when ready
//       for now it is easier to work in a separate file
static NSString * const msl_library_source = @"see metal.metal";

// Here to assist with NSBundle Path Hack
@interface GGMLMetalClass : NSObject
@end
@implementation GGMLMetalClass
@end

struct ggml_metal_context * ggml_metal_init(int n_cb) {
    fprintf(stderr, "%s: allocating\n", __func__);

    struct ggml_metal_context * ctx = malloc(sizeof(struct ggml_metal_context));

    ctx->n_cb   = n_cb;
    ctx->device = MTLCreateSystemDefaultDevice();
    ctx->queue  = [ctx->device newCommandQueue];
    ctx->n_buffers = 0;
    ctx->concur_list_len = 0;

    // determine if we can use MPS
    if (MPSSupportsMTLDevice(ctx->device)) {
        fprintf(stderr, "%s: using MPS\n", __func__);
    } else {
        fprintf(stderr, "%s: not using MPS\n", __func__);
        GGML_ASSERT(false && "MPS not supported");
    }

#if 0
    // compile from source string and show compile log
    {
        NSError * error = nil;

        ctx->library = [ctx->device newLibraryWithSource:msl_library_source options:nil error:&error];
        if (error) {
            fprintf(stderr, "%s: error: %s\n", __func__, [[error description] UTF8String]);
            exit(1);
        }
    }
#else
    UNUSED(msl_library_source);

    // read the source from "ggml-metal.metal" into a string and use newLibraryWithSource
    {
        NSError * error = nil;

        //NSString * path = [[NSBundle mainBundle] pathForResource:@"../../examples/metal/metal" ofType:@"metal"];
        NSBundle * bundle = [NSBundle bundleForClass:[GGMLMetalClass class]];
        NSString * path = [bundle pathForResource:@"ggml-metal" ofType:@"metal"];
        fprintf(stderr, "%s: loading '%s'\n", __func__, [path UTF8String]);

        NSString * src  = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
        if (error) {
            fprintf(stderr, "%s: error: %s\n", __func__, [[error description] UTF8String]);
            exit(1);
        }

#ifdef GGML_QKK_64
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.preprocessorMacros = @{ @"QK_K" : @(64) };
        ctx->library = [ctx->device newLibraryWithSource:src options:options error:&error];
#else
        ctx->library = [ctx->device newLibraryWithSource:src options:nil error:&error];
#endif
        if (error) {
            fprintf(stderr, "%s: error: %s\n", __func__, [[error description] UTF8String]);
            exit(1);
        }
    }
#endif

    // load kernels
    {
#define GGML_METAL_ADD_KERNEL(name) \
        ctx->function_##name = [ctx->library newFunctionWithName:@"kernel_"#name]; \
        ctx->pipeline_##name = [ctx->device newComputePipelineStateWithFunction:ctx->function_##name error:nil]; \
        fprintf(stderr, "%s: loaded %-32s %16p\n", __func__, "kernel_"#name, (void *) ctx->pipeline_##name);

        GGML_METAL_ADD_KERNEL(add);
        GGML_METAL_ADD_KERNEL(add_row);
        GGML_METAL_ADD_KERNEL(mul);
        GGML_METAL_ADD_KERNEL(mul_row);
        GGML_METAL_ADD_KERNEL(scale);
        GGML_METAL_ADD_KERNEL(silu);
        GGML_METAL_ADD_KERNEL(relu);
        GGML_METAL_ADD_KERNEL(gelu);
        GGML_METAL_ADD_KERNEL(soft_max);
        GGML_METAL_ADD_KERNEL(diag_mask_inf);
        GGML_METAL_ADD_KERNEL(get_rows_f16);
        GGML_METAL_ADD_KERNEL(get_rows_q4_0);
        GGML_METAL_ADD_KERNEL(get_rows_q4_1);
        GGML_METAL_ADD_KERNEL(get_rows_q2_K);
        GGML_METAL_ADD_KERNEL(get_rows_q3_K);
        GGML_METAL_ADD_KERNEL(get_rows_q4_K);
        GGML_METAL_ADD_KERNEL(get_rows_q5_K);
        GGML_METAL_ADD_KERNEL(get_rows_q6_K);
        GGML_METAL_ADD_KERNEL(rms_norm);
        GGML_METAL_ADD_KERNEL(norm);
        GGML_METAL_ADD_KERNEL(mul_mat_f16_f32);
        GGML_METAL_ADD_KERNEL(mul_mat_q4_0_f32);
        GGML_METAL_ADD_KERNEL(mul_mat_q4_1_f32);
        GGML_METAL_ADD_KERNEL(mul_mat_q2_K_f32);
        GGML_METAL_ADD_KERNEL(mul_mat_q3_K_f32);
        GGML_METAL_ADD_KERNEL(mul_mat_q4_K_f32);
        GGML_METAL_ADD_KERNEL(mul_mat_q5_K_f32);
        GGML_METAL_ADD_KERNEL(mul_mat_q6_K_f32);
        GGML_METAL_ADD_KERNEL(rope);
        GGML_METAL_ADD_KERNEL(alibi_f32);
        GGML_METAL_ADD_KERNEL(cpy_f32_f16);
        GGML_METAL_ADD_KERNEL(cpy_f32_f32);
        GGML_METAL_ADD_KERNEL(cpy_f16_f16);

#undef GGML_METAL_ADD_KERNEL
    }

    fprintf(stderr, "%s: recommendedMaxWorkingSetSize = %8.2f MB\n", __func__, ctx->device.recommendedMaxWorkingSetSize / 1024.0 / 1024.0);
    fprintf(stderr, "%s: hasUnifiedMemory             = %s\n",       __func__, ctx->device.hasUnifiedMemory ? "true" : "false");
    if (ctx->device.maxTransferRate != 0) {
        fprintf(stderr, "%s: maxTransferRate              = %8.2f MB/s\n", __func__, ctx->device.maxTransferRate / 1024.0 / 1024.0);
    } else {
        fprintf(stderr, "%s: maxTransferRate              = built-in GPU\n", __func__);
    }

    return ctx;
}

void ggml_metal_free(struct ggml_metal_context * ctx) {
    fprintf(stderr, "%s: deallocating\n", __func__);
    for (int i = 0; i < ctx->n_buffers; ++i) {
        [ctx->buffers[i].metal release];
    }
    free(ctx);
}

void ggml_metal_set_n_cb(struct ggml_metal_context * ctx, int n_cb) {
    ctx->n_cb = n_cb;
}

bool ggml_metal_if_optimized(struct ggml_metal_context * ctx) {
    if (ctx->concur_list_len) {
        return true;
    }
    return false;
}

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
static id<MTLBuffer> ggml_metal_get_buffer(struct ggml_metal_context * ctx, struct ggml_tensor * t, size_t * offs) {
    //fprintf(stderr, "%s: data tensor '%16s', offs_data = %8ld, offs_eval = %8ld, offs_cach = %8ld\n", __func__, t->name, offs_data, offs_eval, offs_cach);

    const int64_t tsize = ggml_nbytes(t);

    // find the view that contains the tensor fully
    for (int i = 0; i < ctx->n_buffers; ++i) {
        const int64_t ioffs = (int64_t) t->data - (int64_t) ctx->buffers[i].data;

        if (ioffs >= 0 && ioffs + tsize <= (int64_t) ctx->buffers[i].size) {
            *offs = (size_t) ioffs;

            //fprintf(stderr, "%s: '%s' tensor '%16s', offs = %8ld\n", __func__, ctx->buffers[i].name, t->name, *offs);

            return ctx->buffers[i].metal;
        }
    }

    fprintf(stderr, "%s: error: buffer is nil\n", __func__);

    return nil;
}

bool ggml_metal_add_buffer(
        struct ggml_metal_context * ctx,
                     const char * name,
                           void * data,
                         size_t   size,
                         size_t   max_size) {
    if (ctx->n_buffers >= GGML_METAL_MAX_BUFFERS) {
        fprintf(stderr, "%s: too many buffers\n", __func__);
        return false;
    }

    if (data) {
        // verify that the buffer does not overlap with any of the existing buffers
        for (int i = 0; i < ctx->n_buffers; ++i) {
            const int64_t ioffs = (int64_t) data - (int64_t) ctx->buffers[i].data;

            if (ioffs >= 0 && ioffs < (int64_t) ctx->buffers[i].size) {
                fprintf(stderr, "%s: error: buffer '%s' overlaps with '%s'\n", __func__, name, ctx->buffers[i].name);
                return false;
            }
        }

        const size_t size_page = getpagesize();

        size_t size_aligned = size;
        if ((size_aligned % size_page) != 0) {
            size_aligned += (size_page - (size_aligned % size_page));
        }

        // the buffer fits into the max buffer size allowed by the device
        if (size_aligned <= ctx->device.maxBufferLength) {
            ctx->buffers[ctx->n_buffers].name = name;
            ctx->buffers[ctx->n_buffers].data = data;
            ctx->buffers[ctx->n_buffers].size = size;

            ctx->buffers[ctx->n_buffers].metal = [ctx->device newBufferWithBytesNoCopy:data length:size_aligned options:MTLResourceStorageModeShared deallocator:nil];

            if (ctx->buffers[ctx->n_buffers].metal == nil) {
                fprintf(stderr, "%s: failed to allocate '%-16s' buffer, size = %8.2f MB\n", __func__, name, size_aligned / 1024.0 / 1024.0);
                return false;
            }

            fprintf(stderr, "%s: allocated '%-16s' buffer, size = %8.2f MB", __func__, name, size_aligned / 1024.0 / 1024.0);

            ++ctx->n_buffers;
        } else {
            // this overlap between the views will guarantee that the tensor with the maximum size will fully fit into
            // one of the views
            const size_t size_ovlp = ((max_size + size_page - 1) / size_page + 1) * size_page; // round-up 2 pages just in case
            const size_t size_step = ctx->device.maxBufferLength - size_ovlp;
            const size_t size_view = ctx->device.maxBufferLength;

            for (size_t i = 0; i < size; i += size_step) {
                const size_t size_step_aligned = (i + size_view <= size) ? size_view : (size_aligned - i);

                ctx->buffers[ctx->n_buffers].name = name;
                ctx->buffers[ctx->n_buffers].data = (void *) ((uint8_t *) data + i);
                ctx->buffers[ctx->n_buffers].size = size_step_aligned;

                ctx->buffers[ctx->n_buffers].metal = [ctx->device newBufferWithBytesNoCopy:(void *) ((uint8_t *) data + i) length:size_step_aligned options:MTLResourceStorageModeShared deallocator:nil];

                if (ctx->buffers[ctx->n_buffers].metal == nil) {
                    fprintf(stderr, "%s: failed to allocate '%-16s' buffer, size = %8.2f MB\n", __func__, name, size_step_aligned / 1024.0 / 1024.0);
                    return false;
                }

                fprintf(stderr, "%s: allocated '%-16s' buffer, size = %8.2f MB, offs = %12ld", __func__, name, size_step_aligned / 1024.0 / 1024.0, i);
                if (i + size_step < size) {
                    fprintf(stderr, "\n");
                }

                ++ctx->n_buffers;
            }
        }

        fprintf(stderr, ", (%8.2f / %8.2f)",
                ctx->device.currentAllocatedSize / 1024.0 / 1024.0,
                ctx->device.recommendedMaxWorkingSetSize / 1024.0 / 1024.0);

        if (ctx->device.currentAllocatedSize > ctx->device.recommendedMaxWorkingSetSize) {
            fprintf(stderr, ", warning: current allocated size is greater than the recommended max working set size\n");
        } else {
            fprintf(stderr, "\n");
        }
    }

    return true;
}

void ggml_metal_set_tensor(
        struct ggml_metal_context * ctx,
        struct ggml_tensor * t) {
    metal_printf("%s: set input for tensor '%s'\n", __func__, t->name);

    size_t offs;
    id<MTLBuffer> id_dst = ggml_metal_get_buffer(ctx, t, &offs);

    memcpy((void *) ((uint8_t *) id_dst.contents + offs), t->data, ggml_nbytes(t));
}

void ggml_metal_get_tensor(
        struct ggml_metal_context * ctx,
        struct ggml_tensor * t) {
    metal_printf("%s: extract results for tensor '%s'\n", __func__, t->name);

    size_t offs;
    id<MTLBuffer> id_src = ggml_metal_get_buffer(ctx, t, &offs);

    memcpy(t->data, (void *) ((uint8_t *) id_src.contents + offs), ggml_nbytes(t));
}

void ggml_metal_graph_find_concurrency(
        struct ggml_metal_context * ctx,
        struct ggml_cgraph * gf) {
    int search_depth = gf->n_nodes; //we only find concurrency in this range to avoid wasting too much time
    int nodes_unused[GGML_MAX_NODES];

    for (int i = 0; i < GGML_MAX_NODES; i++) {ctx->concur_list[i] = 0;}
    for (int i = 0; i < gf->n_nodes; i++) {nodes_unused[i] = 1;}
    ctx->concur_list_len = 0;

    int n_left = gf->n_nodes;
    int n_start = 0; // all nodes before n_start at nodes_unused array have been sorted and store back to ctx->concur_list
    int level_pos = 0;  // at ctx->concur_list, the last layer (level) ends at level_pos

    while (n_left > 0) {
        // number of nodes at a layer (that can be issued concurrently)
        int concurrency = 0;
        for (int i = n_start; i < ((n_start + search_depth > gf->n_nodes) ? gf->n_nodes : n_start + search_depth); i++) {
            if (nodes_unused[i]) {
                // if the requirements for gf->nodes[i] are satisfied
                int exe_flag=1;
                // scan all srcs
                for (int src_ind = 0; src_ind < GGML_MAX_SRC; src_ind++) {
                    struct ggml_tensor * src_cur = gf->nodes[i]->src[src_ind];
                    if (src_cur) {
                        // if is leaf nodes it's satisfied.
                        if (src_cur->op == GGML_OP_NONE && src_cur->grad == NULL) {continue;}

                        // otherwise this src should be the output from previous nodes.
                        int is_found = 0;
                        // scan 2*search_depth back because we inserted barrier.
                        for (int j = ((level_pos - 2*search_depth) < 0 ? 0 : (level_pos - 2*search_depth)); j < level_pos; j++) {
                            if (gf->nodes[ctx->concur_list[j]] == src_cur) {is_found = 1; break;}
                        }
                        if (is_found == 0) {exe_flag = 0; break;}
                    }
                }
                if (exe_flag) {
                    // check if nodes[i]'s data will be overwritten by a node before nodes[i].
                    // if node[5] and node[3] write to the same memory region, then we can't issue node[5] before node[3]
                    int64_t data_start = (int64_t) gf->nodes[i]->data;
                    int64_t length = (int64_t) ggml_nbytes(gf->nodes[i]);
                    for (int j = n_start; j < i; j++) {
                        if (nodes_unused[j] && gf->nodes[j]->op != GGML_OP_RESHAPE \
                                            && gf->nodes[j]->op != GGML_OP_VIEW \
                                            && gf->nodes[j]->op != GGML_OP_TRANSPOSE \
                                            && gf->nodes[j]->op != GGML_OP_PERMUTE) {
                            if (((int64_t)gf->nodes[j]->data) >= data_start + length || \
                                ((int64_t)gf->nodes[j]->data) + (int64_t) ggml_nbytes(gf->nodes[j]) <= data_start) {
                                continue;
                            } else {
                                exe_flag = 0;
                            }
                        }
                    }
                }
                if (exe_flag) {
                    ctx->concur_list[level_pos + concurrency] = i;
                    nodes_unused[i] = 0;
                    concurrency++;
                    ctx->concur_list_len++;
                }
            }
        }
        n_left -= concurrency;
        // adding a barrier different layer
        ctx->concur_list[level_pos + concurrency] = -1;
        ctx->concur_list_len++;
        // jump all sorted nodes at nodes_bak
        while (!nodes_unused[n_start]) {n_start++;}
        level_pos += concurrency + 1;
    }

    if (ctx->concur_list_len > GGML_MAX_NODES) {
        fprintf(stderr, "%s: too many elements for metal ctx->concur_list!\n", __func__);
    }
}

void ggml_metal_graph_compute(
        struct ggml_metal_context * ctx,
               struct ggml_cgraph * gf) {
    metal_printf("%s: evaluating graph\n", __func__);

    // if there is ctx->concur_list, dispatch concurrently
    // else fallback to serial dispatch
    MTLComputePassDescriptor * edesc = MTLComputePassDescriptor.computePassDescriptor;

    const bool has_concur = ctx->concur_list_len && ctx->concur_list_len <= GGML_MAX_NODES;

    const int n_nodes  = has_concur ? ctx->concur_list_len      : gf->n_nodes;
    edesc.dispatchType = has_concur ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial;

    // create multiple command buffers and enqueue them
    // then, we encode the graph into the command buffers in parallel

    const int n_cb = ctx->n_cb;

    NSMutableArray * command_buffers = [NSMutableArray arrayWithCapacity:n_cb];

    for (int i = 0; i < n_cb; ++i) {
        command_buffers[i] = [ctx->queue commandBuffer];

        // enqueue the command buffers in order to specify their execution order
        [command_buffers[i] enqueue];
    }

    // TODO: is this the best way to start threads?
    dispatch_queue_t queue = dispatch_queue_create("llama.cpp", DISPATCH_QUEUE_CONCURRENT);

    for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
        const int n_nodes_per_cb = (n_nodes + n_cb - 1) / n_cb;

        dispatch_async(queue, ^{
            size_t offs_src0 = 0;
            size_t offs_src1 = 0;
            size_t offs_dst  = 0;

            id<MTLCommandBuffer> command_buffer = command_buffers[cb_idx];

            id<MTLComputeCommandEncoder> encoder = nil;

            const int node_start =                                  (cb_idx + 0) * n_nodes_per_cb;
            const int node_end   = (cb_idx == n_cb - 1) ? n_nodes : (cb_idx + 1) * n_nodes_per_cb;

            for (int ind = node_start; ind < node_end; ++ind) {
                const int i = has_concur ? ctx->concur_list[ind] : ind;

                if (i == -1) {
                    if (encoder == nil) {
                        encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                        continue;
                    }
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    continue;
                }

                metal_printf("%s: encoding node %3d, op = %8s\n", __func__, i, ggml_op_name(gf->nodes[i]->op));

                struct ggml_tensor * src0 = gf->nodes[i]->src[0];
                struct ggml_tensor * src1 = gf->nodes[i]->src[1];
                struct ggml_tensor * dst  = gf->nodes[i];

                const int64_t  ne00 = src0 ? src0->ne[0] : 0;
                const int64_t  ne01 = src0 ? src0->ne[1] : 0;
                const int64_t  ne02 = src0 ? src0->ne[2] : 0;
                const int64_t  ne03 = src0 ? src0->ne[3] : 0;

                const uint64_t nb00 = src0 ? src0->nb[0] : 0;
                const uint64_t nb01 = src0 ? src0->nb[1] : 0;
                const uint64_t nb02 = src0 ? src0->nb[2] : 0;
                const uint64_t nb03 = src0 ? src0->nb[3] : 0;

                const int64_t  ne10 = src1 ? src1->ne[0] : 0;
                const int64_t  ne11 = src1 ? src1->ne[1] : 0;
                const int64_t  ne12 = src1 ? src1->ne[2] : 0;
                const int64_t  ne13 = src1 ? src1->ne[3] : 0; UNUSED(ne13);

                const uint64_t nb10 = src1 ? src1->nb[0] : 0;
                const uint64_t nb11 = src1 ? src1->nb[1] : 0;
                const uint64_t nb12 = src1 ? src1->nb[2] : 0;
                const uint64_t nb13 = src1 ? src1->nb[3] : 0; UNUSED(nb13);

                const int64_t  ne0  = dst ? dst->ne[0] : 0;
                const int64_t  ne1  = dst ? dst->ne[1] : 0;
                const int64_t  ne2  = dst ? dst->ne[2] : 0;
                const int64_t  ne3  = dst ? dst->ne[3] : 0;

                const uint64_t nb0  = dst ? dst->nb[0] : 0;
                const uint64_t nb1  = dst ? dst->nb[1] : 0;
                const uint64_t nb2  = dst ? dst->nb[2] : 0;
                const uint64_t nb3  = dst ? dst->nb[3] : 0;

                const enum ggml_type src0t = src0 ? src0->type : GGML_TYPE_COUNT;
                const enum ggml_type src1t = src1 ? src1->type : GGML_TYPE_COUNT;
                const enum ggml_type dstt  = dst  ? dst->type  : GGML_TYPE_COUNT;

                id<MTLBuffer> id_src0 = src0 ? ggml_metal_get_buffer(ctx, src0, &offs_src0) : nil;
                id<MTLBuffer> id_src1 = src1 ? ggml_metal_get_buffer(ctx, src1, &offs_src1) : nil;
                id<MTLBuffer> id_dst  = dst  ? ggml_metal_get_buffer(ctx, dst,  &offs_dst)  : nil;

                //metal_printf("%s: op - %s\n", __func__, ggml_op_name(dst->op));
                //if (src0) {
                //    metal_printf("%s: src0 - %4s [%5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(src0t), ne00, ne01, ne02,
                //            ggml_is_contiguous(src0), src0->name);
                //}
                //if (src1) {
                //    metal_printf("%s: src1 - %4s [%5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(src1t), ne10, ne11, ne12,
                //            ggml_is_contiguous(src1), src1->name);
                //}
                //if (dst) {
                //    metal_printf("%s: dst  - %4s [%5lld, %5lld, %5lld], 1, %s\n",  __func__, ggml_type_name(dstt),  ne0,  ne1,  ne2,
                //            dst->name);
                //}

                switch (dst->op) {
                    case GGML_OP_NONE:
                    case GGML_OP_RESHAPE:
                    case GGML_OP_VIEW:
                    case GGML_OP_TRANSPOSE:
                    case GGML_OP_PERMUTE:
                        {
                            // noop
                        } break;
                    case GGML_OP_ADD:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                            }

                            if (ggml_nelements(src1) == ne10) {
                                // src1 is a row
                                [encoder setComputePipelineState:ctx->pipeline_add_row];
                            } else {
                                [encoder setComputePipelineState:ctx->pipeline_add];
                            }
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                            [encoder setBytes:&ne00 length:sizeof(ne00) atIndex:3];

                            const int64_t n = ggml_nelements(dst);

                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case GGML_OP_MUL:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                            }

                            if (ggml_nelements(src1) == ne10) {
                                // src1 is a row
                                [encoder setComputePipelineState:ctx->pipeline_mul_row];
                            } else {
                                [encoder setComputePipelineState:ctx->pipeline_mul];
                            }
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                            [encoder setBytes:&ne00 length:sizeof(ne00) atIndex:3];

                            const int64_t n = ggml_nelements(dst);

                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case GGML_OP_SCALE:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                            }

                            const float scale = *(const float *) src1->data;

                            [encoder setComputePipelineState:ctx->pipeline_scale];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&scale length:sizeof(scale) atIndex:2];

                            const int64_t n = ggml_nelements(dst);

                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case GGML_OP_UNARY:
                        switch (ggml_get_unary_op(gf->nodes[i])) {
                            case GGML_UNARY_OP_SILU:
                                {
                                    if (encoder == nil) {
                                        encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                                    }

                                    [encoder setComputePipelineState:ctx->pipeline_silu];
                                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                                    const int64_t n = ggml_nelements(dst);

                                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                                } break;
                            case GGML_UNARY_OP_RELU:
                                {
                                    if (encoder == nil) {
                                        encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                                    }

                                    [encoder setComputePipelineState:ctx->pipeline_relu];
                                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                                    const int64_t n = ggml_nelements(dst);

                                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                                } break;
                            case GGML_UNARY_OP_GELU:
                                {
                                    if (encoder == nil) {
                                        encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                                    }

                                    [encoder setComputePipelineState:ctx->pipeline_gelu];
                                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                                    const int64_t n = ggml_nelements(dst);

                                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                                } break;
                            default:
                                {
                                    fprintf(stderr, "%s: node %3d, op = %8s not implemented\n", __func__, i, ggml_op_name(dst->op));
                                    GGML_ASSERT(false);
                                }
                        } break;
                    case GGML_OP_SOFT_MAX:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                            }

                            const int nth = 32;

                            [encoder setComputePipelineState:ctx->pipeline_soft_max];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00 length:sizeof(ne00) atIndex:2];
                            [encoder setBytes:&ne01 length:sizeof(ne01) atIndex:3];
                            [encoder setBytes:&ne02 length:sizeof(ne02) atIndex:4];
                            [encoder setThreadgroupMemoryLength:nth*sizeof(float) atIndex:0];

                            [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        } break;
                    case GGML_OP_DIAG_MASK_INF:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                            }

                            const int n_past = ((int32_t *)(dst->op_params))[0];

                            [encoder setComputePipelineState:ctx->pipeline_diag_mask_inf];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00   length:sizeof(ne00) atIndex:2];
                            [encoder setBytes:&ne01   length:sizeof(ne01) atIndex:3];
                            [encoder setBytes:&n_past length:sizeof(int)  atIndex:4];

                            [encoder dispatchThreadgroups:MTLSizeMake(ne00, ne01, ne02) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case GGML_OP_MUL_MAT:
                        {
                            // TODO: needs to be updated after PR: https://github.com/ggerganov/ggml/pull/224

                            GGML_ASSERT(ne00 == ne10);
                            // GGML_ASSERT(ne02 == ne12); // Should be checked on individual data types until broadcast is implemented everywhere
                            GGML_ASSERT(ne03 == ne13);

                            if (ggml_is_contiguous(src0) &&
                                ggml_is_contiguous(src1) &&
                                (src0t == GGML_TYPE_F32 || src0t == GGML_TYPE_F16) && ne11 > 1) {

                                if (encoder != nil) {
                                    [encoder endEncoding];
                                    encoder = nil;
                                }

                                MPSDataType src0dt = src0t == GGML_TYPE_F32 ? MPSDataTypeFloat32 : MPSDataTypeFloat16;
                                MPSDataType src1dt = src1t == GGML_TYPE_F32 ? MPSDataTypeFloat32 : MPSDataTypeFloat16;

                                // for F32 x F32 we use MPS
                                MPSMatrixDescriptor * desc0 = [MPSMatrixDescriptor
                                    matrixDescriptorWithRows:ne01 columns:ne00 rowBytes:src0->nb[1] dataType:src0dt];

                                MPSMatrixDescriptor * desc1 = [MPSMatrixDescriptor
                                    matrixDescriptorWithRows:ne11 columns:ne10 rowBytes:src1->nb[1] dataType:src1dt];

                                MPSMatrixDescriptor * desc  = [MPSMatrixDescriptor
                                    matrixDescriptorWithRows:ne1 columns:ne0 rowBytes:dst->nb[1] dataType:MPSDataTypeFloat32];

                                MPSMatrixMultiplication * mul = [[MPSMatrixMultiplication alloc]
                                    initWithDevice:ctx->device transposeLeft:false transposeRight:true
                                        resultRows:ne11 resultColumns:ne01 interiorColumns:ne00 alpha:1.0 beta:0.0];

                                // we need to do ne12 multiplications
                                // TODO: is there a way to do this in parallel - currently very slow ..
                                // TODO: might be possible to offload part of the computation to ANE using Accelerate's CBLAS
                                for (int64_t i02 = 0; i02 < ne12; ++i02) {
                                    size_t offs_src0_cur = offs_src0 + i02/(ne12/ne02)*nb02; // gqa not used for now
                                    size_t offs_src1_cur = offs_src1 + i02*nb12;
                                    size_t offs_dst_cur  = offs_dst  + i02*nb2;

                                    MPSMatrix * mat_src0 = [[MPSMatrix alloc] initWithBuffer:id_src0 offset:offs_src0_cur descriptor:desc0];
                                    MPSMatrix * mat_src1 = [[MPSMatrix alloc] initWithBuffer:id_src1 offset:offs_src1_cur descriptor:desc1];
                                    MPSMatrix * mat_dst  = [[MPSMatrix alloc] initWithBuffer:id_dst  offset:offs_dst_cur  descriptor:desc ];

                                    [mul encodeToCommandBuffer:command_buffer leftMatrix:mat_src1 rightMatrix:mat_src0 resultMatrix:mat_dst];
                                }
                            } else {
                                if (encoder == nil) {
                                    encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                                }

                                int nth0 = 32;
                                int nth1 = 1;

                                // use custom matrix x vector kernel
                                switch (src0t) {
                                    case GGML_TYPE_F16:
                                        {
                                            nth0 = 64;
                                            nth1 = 1;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_f16_f32];
                                        } break;
                                    case GGML_TYPE_Q4_0:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 8;
                                            nth1 = 8;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q4_0_f32];
                                        } break;
                                    case GGML_TYPE_Q4_1:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 8;
                                            nth1 = 8;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q4_1_f32];
                                        } break;
                                    case GGML_TYPE_Q2_K:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 2;
                                            nth1 = 32;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q2_K_f32];
                                        } break;
                                    case GGML_TYPE_Q3_K:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 2;
                                            nth1 = 32;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q3_K_f32];
                                        } break;
                                    case GGML_TYPE_Q4_K:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 2;
                                            nth1 = 32;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q4_K_f32];
                                        } break;
                                    case GGML_TYPE_Q5_K:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 2;
                                            nth1 = 32;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q5_K_f32];
                                        } break;
                                    case GGML_TYPE_Q6_K:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 2;
                                            nth1 = 32;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q6_K_f32];
                                        } break;
                                    default:
                                        {
                                            fprintf(stderr, "Asserting on type %d\n",(int)src0t);
                                            GGML_ASSERT(false && "not implemented");
                                        }
                                };

                                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                                [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                                [encoder setBytes:&ne00 length:sizeof(ne00) atIndex:3];
                                [encoder setBytes:&ne01 length:sizeof(ne01) atIndex:4];
                                [encoder setBytes:&ne02 length:sizeof(ne02) atIndex:5];
                                [encoder setBytes:&nb00 length:sizeof(nb00) atIndex:6];
                                [encoder setBytes:&nb01 length:sizeof(nb01) atIndex:7];
                                [encoder setBytes:&nb02 length:sizeof(nb02) atIndex:8];
                                [encoder setBytes:&ne10 length:sizeof(ne10) atIndex:9];
                                [encoder setBytes:&ne11 length:sizeof(ne11) atIndex:10];
                                [encoder setBytes:&ne12 length:sizeof(ne12) atIndex:11];
                                [encoder setBytes:&nb10 length:sizeof(nb10) atIndex:12];
                                [encoder setBytes:&nb11 length:sizeof(nb11) atIndex:13];
                                [encoder setBytes:&nb12 length:sizeof(nb12) atIndex:14];
                                [encoder setBytes:&ne0  length:sizeof(ne0)  atIndex:15];
                                [encoder setBytes:&ne1  length:sizeof(ne1)  atIndex:16];

                                if (src0t == GGML_TYPE_Q4_0 || src0t == GGML_TYPE_Q4_1 ||
                                    src0t == GGML_TYPE_Q2_K || src0t == GGML_TYPE_Q4_K) {
                                    [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 7) / 8, ne11, 1) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                                }
                                else if (src0t == GGML_TYPE_Q3_K) {
#ifdef GGML_QKK_64
                                    [encoder dispatchThreadgroups:MTLSizeMake((ne01+1)/2, ne11, 1) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
#else
                                    [encoder dispatchThreadgroups:MTLSizeMake((ne01+3)/4, ne11, 1) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
#endif
                                }
                                else if (src0t == GGML_TYPE_Q5_K) {
                                    [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 3) / 4, ne11, 1) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                                }
                                else if (src0t == GGML_TYPE_Q6_K) {
                                    [encoder dispatchThreadgroups:MTLSizeMake((ne01+1)/2, ne11, 1) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                                } else {
                                    [encoder setThreadgroupMemoryLength:nth0*sizeof(float) atIndex:0];
                                    [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne11, ne12) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                                }
                            }
                        } break;
                    case GGML_OP_GET_ROWS:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                            }

                            switch (src0->type) {
                                case GGML_TYPE_F16:  [encoder setComputePipelineState:ctx->pipeline_get_rows_f16]; break;
                                case GGML_TYPE_Q4_0: [encoder setComputePipelineState:ctx->pipeline_get_rows_q4_0]; break;
                                case GGML_TYPE_Q4_1: [encoder setComputePipelineState:ctx->pipeline_get_rows_q4_1]; break;
                                case GGML_TYPE_Q2_K: [encoder setComputePipelineState:ctx->pipeline_get_rows_q2_K]; break;
                                case GGML_TYPE_Q3_K: [encoder setComputePipelineState:ctx->pipeline_get_rows_q3_K]; break;
                                case GGML_TYPE_Q4_K: [encoder setComputePipelineState:ctx->pipeline_get_rows_q4_K]; break;
                                case GGML_TYPE_Q5_K: [encoder setComputePipelineState:ctx->pipeline_get_rows_q5_K]; break;
                                case GGML_TYPE_Q6_K: [encoder setComputePipelineState:ctx->pipeline_get_rows_q6_K]; break;
                                default: GGML_ASSERT(false && "not implemented");
                            }

                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                            [encoder setBytes:&(src0->ne[0]) length:sizeof( int64_t) atIndex:3];
                            [encoder setBytes:&(src0->nb[1]) length:sizeof(uint64_t) atIndex:4];
                            [encoder setBytes:&(dst->nb[1])  length:sizeof(uint64_t) atIndex:5];

                            const int64_t n = ggml_nelements(src1);

                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case GGML_OP_RMS_NORM:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                            }

                            float eps;
                            memcpy(&eps, dst->op_params, sizeof(float));

                            const int nth = 512;

                            [encoder setComputePipelineState:ctx->pipeline_rms_norm];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00 length:sizeof( int64_t) atIndex:2];
                            [encoder setBytes:&nb01 length:sizeof(uint64_t) atIndex:3];
                            [encoder setBytes:&eps  length:sizeof(   float) atIndex:4];
                            [encoder setThreadgroupMemoryLength:nth/32*sizeof(float) atIndex:0];

                            const int64_t nrows = ggml_nrows(src0);

                            [encoder dispatchThreadgroups:MTLSizeMake(nrows, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        } break;
                    case GGML_OP_NORM:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                            }

                            const float eps = 1e-5f;

                            const int nth = 256;

                            [encoder setComputePipelineState:ctx->pipeline_norm];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00 length:sizeof( int64_t) atIndex:2];
                            [encoder setBytes:&nb01 length:sizeof(uint64_t) atIndex:3];
                            [encoder setBytes:&eps  length:sizeof(   float) atIndex:4];
                            [encoder setThreadgroupMemoryLength:nth*sizeof(float) atIndex:0];

                            const int64_t nrows = ggml_nrows(src0);

                            [encoder dispatchThreadgroups:MTLSizeMake(nrows, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        } break;
                    case GGML_OP_ALIBI:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                            }

                            GGML_ASSERT((src0t == GGML_TYPE_F32));

                            const int n_past = ((int32_t *) dst->op_params)[0]; UNUSED(n_past);
                            const int n_head = ((int32_t *) dst->op_params)[1];
                            float max_bias;
                            memcpy(&max_bias, (int32_t *) dst->op_params + 2, sizeof(float));

                            if (__builtin_popcount(n_head) != 1) {
                                GGML_ASSERT(false && "only power-of-two n_head implemented");
                            }

                            const int n_heads_log2_floor = 1 << (int) floor(log2(n_head));
                            const float m0 = powf(2.0f, -(max_bias) / n_heads_log2_floor);

                            [encoder setComputePipelineState:ctx->pipeline_alibi_f32];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00 length:sizeof( int64_t) atIndex:2];
                            [encoder setBytes:&ne01 length:sizeof( int64_t) atIndex:3];
                            [encoder setBytes:&ne02 length:sizeof( int64_t) atIndex:4];
                            [encoder setBytes:&ne03 length:sizeof( int64_t) atIndex:5];
                            [encoder setBytes:&nb00 length:sizeof(uint64_t) atIndex:6];
                            [encoder setBytes:&nb01 length:sizeof(uint64_t) atIndex:7];
                            [encoder setBytes:&nb02 length:sizeof(uint64_t) atIndex:8];
                            [encoder setBytes:&nb03 length:sizeof(uint64_t) atIndex:9];
                            [encoder setBytes:&ne0  length:sizeof( int64_t) atIndex:10];
                            [encoder setBytes:&ne1  length:sizeof( int64_t) atIndex:11];
                            [encoder setBytes:&ne2  length:sizeof( int64_t) atIndex:12];
                            [encoder setBytes:&ne3  length:sizeof( int64_t) atIndex:13];
                            [encoder setBytes:&nb0  length:sizeof(uint64_t) atIndex:14];
                            [encoder setBytes:&nb1  length:sizeof(uint64_t) atIndex:15];
                            [encoder setBytes:&nb2  length:sizeof(uint64_t) atIndex:16];
                            [encoder setBytes:&nb3  length:sizeof(uint64_t) atIndex:17];
                            [encoder setBytes:&m0  length:sizeof(    float) atIndex:18];
                            const int nth = 32;
                            [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        } break;
                    case GGML_OP_ROPE:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                            }

                            const int n_past = ((int32_t *) dst->op_params)[0];
                            const int n_dims = ((int32_t *) dst->op_params)[1];
                            const int mode   = ((int32_t *) dst->op_params)[2];

                            float freq_base;
                            float freq_scale;
                            memcpy(&freq_base,  (int32_t *) dst->op_params + 4, sizeof(float));
                            memcpy(&freq_scale, (int32_t *) dst->op_params + 5, sizeof(float));

                            [encoder setComputePipelineState:ctx->pipeline_rope];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00    length:sizeof( int64_t) atIndex:2];
                            [encoder setBytes:&ne01    length:sizeof( int64_t) atIndex:3];
                            [encoder setBytes:&ne02    length:sizeof( int64_t) atIndex:4];
                            [encoder setBytes:&ne03    length:sizeof( int64_t) atIndex:5];
                            [encoder setBytes:&nb00    length:sizeof(uint64_t) atIndex:6];
                            [encoder setBytes:&nb01    length:sizeof(uint64_t) atIndex:7];
                            [encoder setBytes:&nb02    length:sizeof(uint64_t) atIndex:8];
                            [encoder setBytes:&nb03    length:sizeof(uint64_t) atIndex:9];
                            [encoder setBytes:&ne0     length:sizeof( int64_t) atIndex:10];
                            [encoder setBytes:&ne1     length:sizeof( int64_t) atIndex:11];
                            [encoder setBytes:&ne2     length:sizeof( int64_t) atIndex:12];
                            [encoder setBytes:&ne3     length:sizeof( int64_t) atIndex:13];
                            [encoder setBytes:&nb0     length:sizeof(uint64_t) atIndex:14];
                            [encoder setBytes:&nb1     length:sizeof(uint64_t) atIndex:15];
                            [encoder setBytes:&nb2     length:sizeof(uint64_t) atIndex:16];
                            [encoder setBytes:&nb3     length:sizeof(uint64_t) atIndex:17];
                            [encoder setBytes:&n_past  length:sizeof(     int) atIndex:18];
                            [encoder setBytes:&n_dims  length:sizeof(     int) atIndex:19];
                            [encoder setBytes:&mode    length:sizeof(     int) atIndex:20];
                            [encoder setBytes:&freq_base  length:sizeof(float) atIndex:21];
                            [encoder setBytes:&freq_scale length:sizeof(float) atIndex:22];

                            [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case GGML_OP_DUP:
                    case GGML_OP_CPY:
                    case GGML_OP_CONT:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                            }

                            const int nth = 32;

                            switch (src0t) {
                                case GGML_TYPE_F32:
                                    {
                                        switch (dstt) {
                                            case GGML_TYPE_F16: [encoder setComputePipelineState:ctx->pipeline_cpy_f32_f16]; break;
                                            case GGML_TYPE_F32: [encoder setComputePipelineState:ctx->pipeline_cpy_f32_f32]; break;
                                            default: GGML_ASSERT(false && "not implemented");
                                        };
                                    } break;
                                case GGML_TYPE_F16:
                                    {
                                        switch (dstt) {
                                            case GGML_TYPE_F16: [encoder setComputePipelineState:ctx->pipeline_cpy_f16_f16]; break;
                                            case GGML_TYPE_F32: GGML_ASSERT(false && "cpy_f16_f32 not implemented"); break;
                                            default: GGML_ASSERT(false && "not implemented");
                                        };
                                    } break;
                                default: GGML_ASSERT(false && "not implemented");
                            }

                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00 length:sizeof( int64_t) atIndex:2];
                            [encoder setBytes:&ne01 length:sizeof( int64_t) atIndex:3];
                            [encoder setBytes:&ne02 length:sizeof( int64_t) atIndex:4];
                            [encoder setBytes:&ne03 length:sizeof( int64_t) atIndex:5];
                            [encoder setBytes:&nb00 length:sizeof(uint64_t) atIndex:6];
                            [encoder setBytes:&nb01 length:sizeof(uint64_t) atIndex:7];
                            [encoder setBytes:&nb02 length:sizeof(uint64_t) atIndex:8];
                            [encoder setBytes:&nb03 length:sizeof(uint64_t) atIndex:9];
                            [encoder setBytes:&ne0  length:sizeof( int64_t) atIndex:10];
                            [encoder setBytes:&ne1  length:sizeof( int64_t) atIndex:11];
                            [encoder setBytes:&ne2  length:sizeof( int64_t) atIndex:12];
                            [encoder setBytes:&ne3  length:sizeof( int64_t) atIndex:13];
                            [encoder setBytes:&nb0  length:sizeof(uint64_t) atIndex:14];
                            [encoder setBytes:&nb1  length:sizeof(uint64_t) atIndex:15];
                            [encoder setBytes:&nb2  length:sizeof(uint64_t) atIndex:16];
                            [encoder setBytes:&nb3  length:sizeof(uint64_t) atIndex:17];

                            [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        } break;
                    default:
                        {
                            fprintf(stderr, "%s: node %3d, op = %8s not implemented\n", __func__, i, ggml_op_name(dst->op));
                            GGML_ASSERT(false);
                        }
                }
            }

            if (encoder != nil) {
                [encoder endEncoding];
                encoder = nil;
            }

            [command_buffer commit];
        });
    }

    // wait for all threads to finish
    dispatch_barrier_sync(queue, ^{});

    [command_buffers[n_cb - 1] waitUntilCompleted];

    // check status of command buffers
    // needed to detect if the device ran out-of-memory for example (#1881)
    for (int i = 0; i < n_cb; i++) {
        MTLCommandBufferStatus status = (MTLCommandBufferStatus) [command_buffers[i] status];
        if (status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "%s: command buffer %d failed with status %lu\n", __func__, i, status);
            GGML_ASSERT(false);
        }
    }
}
