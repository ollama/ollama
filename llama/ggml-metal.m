/**
 * llama.cpp - git 5bf2a2771886ee86137e01dbc7492f78fb392066
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

    // custom kernels
#define GGML_METAL_DECL_KERNEL(name) \
    id<MTLFunction>             function_##name; \
    id<MTLComputePipelineState> pipeline_##name

    GGML_METAL_DECL_KERNEL(add);
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

void ggml_metal_graph_compute(
        struct ggml_metal_context * ctx,
               struct ggml_cgraph * gf) {
    metal_printf("%s: evaluating graph\n", __func__);

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
        const int n_nodes_per_cb = (gf->n_nodes + n_cb - 1) / n_cb;

        dispatch_async(queue, ^{
            size_t offs_src0 = 0;
            size_t offs_src1 = 0;
            size_t offs_dst  = 0;

            id<MTLCommandBuffer> command_buffer = command_buffers[cb_idx];

            id<MTLComputeCommandEncoder> encoder = nil;

            const int node_start =                                      (cb_idx + 0) * n_nodes_per_cb;
            const int node_end   = (cb_idx == n_cb - 1) ? gf->n_nodes : (cb_idx + 1) * n_nodes_per_cb;

            for (int i = node_start; i < node_end; ++i) {
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
                                encoder = [command_buffer computeCommandEncoder];
                            }

                            [encoder setComputePipelineState:ctx->pipeline_add];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];

                            const int64_t n = ggml_nelements(dst);

                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case GGML_OP_MUL:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoder];
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
                                encoder = [command_buffer computeCommandEncoder];
                            }

                            const float scale = *(const float *) src1->data;

                            [encoder setComputePipelineState:ctx->pipeline_scale];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&scale length:sizeof(scale) atIndex:2];

                            const int64_t n = ggml_nelements(dst);

                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case GGML_OP_SILU:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoder];
                            }

                            [encoder setComputePipelineState:ctx->pipeline_silu];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                            const int64_t n = ggml_nelements(dst);

                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case GGML_OP_RELU:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoder];
                            }

                            [encoder setComputePipelineState:ctx->pipeline_relu];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                            const int64_t n = ggml_nelements(dst);

                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case GGML_OP_GELU:
                    {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoder];
                            }

                            [encoder setComputePipelineState:ctx->pipeline_gelu];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                            const int64_t n = ggml_nelements(dst);

                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                    } break;
                    case GGML_OP_SOFT_MAX:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoder];
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
                                encoder = [command_buffer computeCommandEncoder];
                            }

                            const int n_past = ((int32_t *)(src1->data))[0];

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
                            GGML_ASSERT(ne02 == ne12);

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

                                // we need to do ne02 multiplications
                                // TODO: is there a way to do this in parallel - currently very slow ..
                                // TODO: might be possible to offload part of the computation to ANE using Accelerate's CBLAS
                                for (int64_t i02 = 0; i02 < ne02; ++i02) {
                                    size_t offs_src0_cur = offs_src0 + i02*nb02;
                                    size_t offs_src1_cur = offs_src1 + i02*nb12;
                                    size_t offs_dst_cur  = offs_dst  + i02*nb2;

                                    MPSMatrix * mat_src0 = [[MPSMatrix alloc] initWithBuffer:id_src0 offset:offs_src0_cur descriptor:desc0];
                                    MPSMatrix * mat_src1 = [[MPSMatrix alloc] initWithBuffer:id_src1 offset:offs_src1_cur descriptor:desc1];
                                    MPSMatrix * mat_dst  = [[MPSMatrix alloc] initWithBuffer:id_dst  offset:offs_dst_cur  descriptor:desc ];

                                    [mul encodeToCommandBuffer:command_buffer leftMatrix:mat_src1 rightMatrix:mat_src0 resultMatrix:mat_dst];
                                }
                            } else {
                                if (encoder == nil) {
                                    encoder = [command_buffer computeCommandEncoder];
                                }

                                int nth0 = 32;
                                int nth1 = 1;

                                // use custom matrix x vector kernel
                                switch (src0t) {
                                    case GGML_TYPE_F16:
                                        {
                                            GGML_ASSERT(ne02 == ne12);

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

                                            nth0 = 4;
                                            nth1 = 16;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q2_K_f32];
                                        } break;
                                    case GGML_TYPE_Q3_K:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 4;
                                            nth1 = 16;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q3_K_f32];
                                        } break;
                                    case GGML_TYPE_Q4_K:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 4;
                                            nth1 = 16;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q4_K_f32];
                                        } break;
                                    case GGML_TYPE_Q5_K:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 4;
                                            nth1 = 16;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q5_K_f32];
                                        } break;
                                    case GGML_TYPE_Q6_K:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 4;
                                            nth1 = 16;
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
                                [encoder setBytes:&nb00 length:sizeof(nb00) atIndex:5];
                                [encoder setBytes:&nb01 length:sizeof(nb01) atIndex:6];
                                [encoder setBytes:&nb02 length:sizeof(nb02) atIndex:7];
                                [encoder setBytes:&ne10 length:sizeof(ne10) atIndex:8];
                                [encoder setBytes:&ne11 length:sizeof(ne11) atIndex:9];
                                [encoder setBytes:&nb10 length:sizeof(nb10) atIndex:10];
                                [encoder setBytes:&nb11 length:sizeof(nb11) atIndex:11];
                                [encoder setBytes:&nb12 length:sizeof(nb12) atIndex:12];
                                [encoder setBytes:&ne0  length:sizeof(ne0)  atIndex:13];
                                [encoder setBytes:&ne1  length:sizeof(ne1)  atIndex:14];

                                if (src0t == GGML_TYPE_Q4_0 || src0t == GGML_TYPE_Q4_1) {
                                    [encoder setThreadgroupMemoryLength:nth0*nth1*sizeof(float) atIndex:0];
                                    [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne11, 1) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                                }
                                else if (src0t == GGML_TYPE_Q2_K ||
                                         src0t == GGML_TYPE_Q3_K ||
                                         src0t == GGML_TYPE_Q4_K ||
                                         src0t == GGML_TYPE_Q5_K ||
                                         src0t == GGML_TYPE_Q6_K) {
                                    [encoder setThreadgroupMemoryLength:nth0*nth1*sizeof(float) atIndex:0];
                                    [encoder dispatchThreadgroups:MTLSizeMake(ne01, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                                } else {
                                    [encoder setThreadgroupMemoryLength:nth0*sizeof(float) atIndex:0];
                                    [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne11, ne12) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                                }
                            }
                        } break;
                    case GGML_OP_GET_ROWS:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoder];
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
                                encoder = [command_buffer computeCommandEncoder];
                            }

                            const float eps = 1e-6f;

                            const int nth = 256;

                            [encoder setComputePipelineState:ctx->pipeline_rms_norm];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00 length:sizeof( int64_t) atIndex:2];
                            [encoder setBytes:&nb01 length:sizeof(uint64_t) atIndex:3];
                            [encoder setBytes:&eps  length:sizeof(   float) atIndex:4];
                            [encoder setThreadgroupMemoryLength:nth*sizeof(float) atIndex:0];

                            const int64_t nrows = ggml_nrows(src0);

                            [encoder dispatchThreadgroups:MTLSizeMake(nrows, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        } break;
                    case GGML_OP_NORM:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoder];
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
                                encoder = [command_buffer computeCommandEncoder];
                            }

                            GGML_ASSERT((src0t == GGML_TYPE_F32));

                            const int   n_past   = ((int32_t *) src1->data)[0]; UNUSED(n_past);
                            const int   n_head   = ((int32_t *) src1->data)[1];
                            const float max_bias = ((float *)   src1->data)[2];

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
                                encoder = [command_buffer computeCommandEncoder];
                            }

                            const int n_dims = ((int32_t *) src1->data)[1];
                            const int mode   = ((int32_t *) src1->data)[2];

                            const int n_past = ((int32_t *)(src1->data))[0];

                            [encoder setComputePipelineState:ctx->pipeline_rope];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00   length:sizeof( int64_t) atIndex:2];
                            [encoder setBytes:&ne01   length:sizeof( int64_t) atIndex:3];
                            [encoder setBytes:&ne02   length:sizeof( int64_t) atIndex:4];
                            [encoder setBytes:&ne03   length:sizeof( int64_t) atIndex:5];
                            [encoder setBytes:&nb00   length:sizeof(uint64_t) atIndex:6];
                            [encoder setBytes:&nb01   length:sizeof(uint64_t) atIndex:7];
                            [encoder setBytes:&nb02   length:sizeof(uint64_t) atIndex:8];
                            [encoder setBytes:&nb03   length:sizeof(uint64_t) atIndex:9];
                            [encoder setBytes:&ne0    length:sizeof( int64_t) atIndex:10];
                            [encoder setBytes:&ne1    length:sizeof( int64_t) atIndex:11];
                            [encoder setBytes:&ne2    length:sizeof( int64_t) atIndex:12];
                            [encoder setBytes:&ne3    length:sizeof( int64_t) atIndex:13];
                            [encoder setBytes:&nb0    length:sizeof(uint64_t) atIndex:14];
                            [encoder setBytes:&nb1    length:sizeof(uint64_t) atIndex:15];
                            [encoder setBytes:&nb2    length:sizeof(uint64_t) atIndex:16];
                            [encoder setBytes:&nb3    length:sizeof(uint64_t) atIndex:17];
                            [encoder setBytes:&n_past length:sizeof(     int) atIndex:18];
                            [encoder setBytes:&n_dims length:sizeof(     int) atIndex:19];
                            [encoder setBytes:&mode   length:sizeof(     int) atIndex:20];

                            [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case GGML_OP_CPY:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoder];
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
                        fprintf(stderr, "%s: node %3d, op = %8s not implemented\n", __func__, i, ggml_op_name(dst->op));
                        GGML_ASSERT(false);
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
