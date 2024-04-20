#import "ggml-metal.h"

#import "ggml-backend-impl.h"
#import "ggml.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#ifdef GGML_METAL_NDEBUG
#define GGML_METAL_LOG_INFO(...)
#define GGML_METAL_LOG_WARN(...)
#define GGML_METAL_LOG_ERROR(...)
#else
#define GGML_METAL_LOG_INFO(...)  ggml_metal_log(GGML_LOG_LEVEL_INFO, __VA_ARGS__)
#define GGML_METAL_LOG_WARN(...)  ggml_metal_log(GGML_LOG_LEVEL_WARN, __VA_ARGS__)
#define GGML_METAL_LOG_ERROR(...) ggml_metal_log(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#endif

#define UNUSED(x) (void)(x)

struct ggml_metal_kernel {
    id<MTLComputePipelineState> pipeline;
};

enum ggml_metal_kernel_type {
    GGML_METAL_KERNEL_TYPE_ADD,
    GGML_METAL_KERNEL_TYPE_ADD_ROW,
    GGML_METAL_KERNEL_TYPE_MUL,
    GGML_METAL_KERNEL_TYPE_MUL_ROW,
    GGML_METAL_KERNEL_TYPE_DIV,
    GGML_METAL_KERNEL_TYPE_DIV_ROW,
    GGML_METAL_KERNEL_TYPE_SCALE,
    GGML_METAL_KERNEL_TYPE_SCALE_4,
    GGML_METAL_KERNEL_TYPE_CLAMP,
    GGML_METAL_KERNEL_TYPE_TANH,
    GGML_METAL_KERNEL_TYPE_RELU,
    GGML_METAL_KERNEL_TYPE_GELU,
    GGML_METAL_KERNEL_TYPE_GELU_4,
    GGML_METAL_KERNEL_TYPE_GELU_QUICK,
    GGML_METAL_KERNEL_TYPE_GELU_QUICK_4,
    GGML_METAL_KERNEL_TYPE_SILU,
    GGML_METAL_KERNEL_TYPE_SILU_4,
    GGML_METAL_KERNEL_TYPE_SOFT_MAX,
    GGML_METAL_KERNEL_TYPE_SOFT_MAX_4,
    GGML_METAL_KERNEL_TYPE_DIAG_MASK_INF,
    GGML_METAL_KERNEL_TYPE_DIAG_MASK_INF_8,
    GGML_METAL_KERNEL_TYPE_GET_ROWS_F32,
    GGML_METAL_KERNEL_TYPE_GET_ROWS_F16,
    GGML_METAL_KERNEL_TYPE_GET_ROWS_Q4_0,
    GGML_METAL_KERNEL_TYPE_GET_ROWS_Q4_1,
    GGML_METAL_KERNEL_TYPE_GET_ROWS_Q5_0,
    GGML_METAL_KERNEL_TYPE_GET_ROWS_Q5_1,
    GGML_METAL_KERNEL_TYPE_GET_ROWS_Q8_0,
    GGML_METAL_KERNEL_TYPE_GET_ROWS_Q2_K,
    GGML_METAL_KERNEL_TYPE_GET_ROWS_Q3_K,
    GGML_METAL_KERNEL_TYPE_GET_ROWS_Q4_K,
    GGML_METAL_KERNEL_TYPE_GET_ROWS_Q5_K,
    GGML_METAL_KERNEL_TYPE_GET_ROWS_Q6_K,
    GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ2_XXS,
    GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ2_XS,
    GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ3_XXS,
    GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ3_S,
    GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ2_S,
    GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ1_S,
    GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ1_M,
    GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ4_NL,
    GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ4_XS,
    GGML_METAL_KERNEL_TYPE_GET_ROWS_I32,
    GGML_METAL_KERNEL_TYPE_RMS_NORM,
    GGML_METAL_KERNEL_TYPE_GROUP_NORM,
    GGML_METAL_KERNEL_TYPE_NORM,
    GGML_METAL_KERNEL_TYPE_MUL_MV_F32_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F16,
    GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F32_1ROW,
    GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F32_L4,
    GGML_METAL_KERNEL_TYPE_MUL_MV_Q4_0_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_Q4_1_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_Q5_0_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_Q5_1_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_Q8_0_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_Q2_K_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_Q3_K_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_Q4_K_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_Q5_K_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_Q6_K_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_IQ2_XXS_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_IQ2_XS_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_IQ3_XXS_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_IQ3_S_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_IQ2_S_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_IQ1_S_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_IQ1_M_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_IQ4_NL_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_IQ4_XS_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F32_F32,
  //GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F16_F16,
    GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F16_F32,
  //GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F16_F32_1ROW,
  //GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F16_F32_L4,
    GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q4_0_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q4_1_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q5_0_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q5_1_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q8_0_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q2_K_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q3_K_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q4_K_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q5_K_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q6_K_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ2_XXS_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ2_XS_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ3_XXS_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ3_S_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ2_S_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ1_S_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ1_M_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ4_NL_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ4_XS_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_F32_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_F16_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_Q4_0_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_Q4_1_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_Q5_0_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_Q5_1_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_Q8_0_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_Q2_K_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_Q3_K_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_Q4_K_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_Q5_K_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_Q6_K_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_IQ2_XXS_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_IQ2_XS_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_IQ3_XXS_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_IQ3_S_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_IQ2_S_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_IQ1_S_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_IQ1_M_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_IQ4_NL_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_IQ4_XS_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_ID_F32_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_ID_F16_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q4_0_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q4_1_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q5_0_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q5_1_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q8_0_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q2_K_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q3_K_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q4_K_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q5_K_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q6_K_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ2_XXS_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ2_XS_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ3_XXS_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ3_S_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ2_S_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ1_S_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ1_M_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ4_NL_F32,
    GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ4_XS_F32,
    GGML_METAL_KERNEL_TYPE_ROPE_F32,
    GGML_METAL_KERNEL_TYPE_ROPE_F16,
    GGML_METAL_KERNEL_TYPE_ALIBI_F32,
    GGML_METAL_KERNEL_TYPE_IM2COL_F16,
    GGML_METAL_KERNEL_TYPE_IM2COL_F32,
    GGML_METAL_KERNEL_TYPE_UPSCALE_F32,
    GGML_METAL_KERNEL_TYPE_PAD_F32,
    GGML_METAL_KERNEL_TYPE_ARANGE_F32,
    GGML_METAL_KERNEL_TYPE_TIMESTEP_EMBEDDING_F32,
    GGML_METAL_KERNEL_TYPE_ARGSORT_F32_I32_ASC,
    GGML_METAL_KERNEL_TYPE_ARGSORT_F32_I32_DESC,
    GGML_METAL_KERNEL_TYPE_LEAKY_RELU_F32,
    GGML_METAL_KERNEL_TYPE_CPY_F32_F16,
    GGML_METAL_KERNEL_TYPE_CPY_F32_F32,
    GGML_METAL_KERNEL_TYPE_CPY_F32_Q8_0,
    GGML_METAL_KERNEL_TYPE_CPY_F32_Q4_0,
    GGML_METAL_KERNEL_TYPE_CPY_F32_Q4_1,
    GGML_METAL_KERNEL_TYPE_CPY_F32_Q5_0,
    GGML_METAL_KERNEL_TYPE_CPY_F32_Q5_1,
    GGML_METAL_KERNEL_TYPE_CPY_F32_IQ4_NL,
    GGML_METAL_KERNEL_TYPE_CPY_F16_F16,
    GGML_METAL_KERNEL_TYPE_CPY_F16_F32,
    GGML_METAL_KERNEL_TYPE_CONCAT,
    GGML_METAL_KERNEL_TYPE_SQR,
    GGML_METAL_KERNEL_TYPE_SUM_ROWS,

    GGML_METAL_KERNEL_TYPE_COUNT
};

struct ggml_metal_context {
    int n_cb;

    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;

    dispatch_queue_t d_queue;

    struct ggml_metal_kernel kernels[GGML_METAL_KERNEL_TYPE_COUNT];

    bool support_simdgroup_reduction;
    bool support_simdgroup_mm;

    bool should_capture_next_compute;
};

// MSL code
// TODO: move the contents here when ready
//       for now it is easier to work in a separate file
// static NSString * const msl_library_source = @"see metal.metal";

// Here to assist with NSBundle Path Hack
@interface GGMLMetalClass : NSObject
@end
@implementation GGMLMetalClass
@end

static void ggml_metal_default_log_callback(enum ggml_log_level level, const char * msg, void * user_data) {
    fprintf(stderr, "%s", msg);

    UNUSED(level);
    UNUSED(user_data);
}

ggml_log_callback ggml_metal_log_callback = ggml_metal_default_log_callback;
void * ggml_metal_log_user_data = NULL;

GGML_ATTRIBUTE_FORMAT(2, 3)
static void ggml_metal_log(enum ggml_log_level level, const char * format, ...){
    if (ggml_metal_log_callback != NULL) {
        va_list args;
        va_start(args, format);
        char buffer[128];
        int len = vsnprintf(buffer, 128, format, args);
        if (len < 128) {
            ggml_metal_log_callback(level, buffer, ggml_metal_log_user_data);
        } else {
            char* buffer2 = malloc(len+1);
            va_end(args);
            va_start(args, format);
            vsnprintf(buffer2, len+1, format, args);
            buffer2[len] = 0;
            ggml_metal_log_callback(level, buffer2, ggml_metal_log_user_data);
            free(buffer2);
        }
        va_end(args);
    }
}

static void * ggml_metal_host_malloc(size_t n) {
    void * data = NULL;
    const int result = posix_memalign((void **) &data, sysconf(_SC_PAGESIZE), n);
    if (result != 0) {
        GGML_METAL_LOG_ERROR("%s: error: posix_memalign failed\n", __func__);
        return NULL;
    }

    return data;
}

static struct ggml_metal_context * ggml_metal_init(int n_cb) {
    GGML_METAL_LOG_INFO("%s: allocating\n", __func__);

#if TARGET_OS_OSX && !GGML_METAL_NDEBUG
    // Show all the Metal device instances in the system
    NSArray * devices = MTLCopyAllDevices();
    for (id<MTLDevice> device in devices) {
        GGML_METAL_LOG_INFO("%s: found device: %s\n", __func__, [[device name] UTF8String]);
    }
    [devices release]; // since it was created by a *Copy* C method
#endif

    // Pick and show default Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    GGML_METAL_LOG_INFO("%s: picking default device: %s\n", __func__, [[device name] UTF8String]);

    // Configure context
    struct ggml_metal_context * ctx = malloc(sizeof(struct ggml_metal_context));
    ctx->device = device;
    ctx->n_cb   = MIN(n_cb, GGML_METAL_MAX_BUFFERS);
    ctx->queue  = [ctx->device newCommandQueue];
    ctx->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

    id<MTLLibrary> metal_library;

    // load library
    //
    // - first check if the library is embedded
    // - then check if the library is in the bundle
    // - if not found, load the source and compile it
    // - if that fails, return NULL
    {
        NSBundle * bundle = nil;
#ifdef SWIFT_PACKAGE
        bundle = SWIFTPM_MODULE_BUNDLE;
#else
        bundle = [NSBundle bundleForClass:[GGMLMetalClass class]];
#endif

        NSError * error = nil;

#if GGML_METAL_EMBED_LIBRARY
        const bool try_metallib = false;
#else
        const bool try_metallib = true;
#endif

        NSString * path_lib = [bundle pathForResource:@"default" ofType:@"metallib"];
        if (try_metallib && path_lib != nil) {
            // pre-compiled library found
            NSURL * libURL = [NSURL fileURLWithPath:path_lib];
            GGML_METAL_LOG_INFO("%s: loading '%s'\n", __func__, [path_lib UTF8String]);

            metal_library = [ctx->device newLibraryWithURL:libURL error:&error];
            if (error) {
                GGML_METAL_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                return NULL;
            }
        } else {
#if GGML_METAL_EMBED_LIBRARY
            GGML_METAL_LOG_INFO("%s: using embedded metal library\n", __func__);

            extern const char ggml_metallib_start[];
            extern const char ggml_metallib_end[];

            NSString * src = [[NSString alloc] initWithBytes:ggml_metallib_start length:(ggml_metallib_end-ggml_metallib_start) encoding:NSUTF8StringEncoding];
#else
            GGML_METAL_LOG_INFO("%s: default.metallib not found, loading from source\n", __func__);

            NSString * path_source;
            NSString * path_resource = [[NSProcessInfo processInfo].environment objectForKey:@"GGML_METAL_PATH_RESOURCES"];

            GGML_METAL_LOG_INFO("%s: GGML_METAL_PATH_RESOURCES = %s\n", __func__, path_resource ? [path_resource UTF8String] : "nil");

            if (path_resource) {
                path_source = [path_resource stringByAppendingPathComponent:@"ggml-metal.metal"];
            } else {
                path_source = [bundle pathForResource:@"ggml-metal" ofType:@"metal"];
            }

            if (path_source == nil) {
                GGML_METAL_LOG_WARN("%s: error: could not use bundle path to find ggml-metal.metal, falling back to trying cwd\n", __func__);
                path_source = @"ggml-metal.metal";
            }

            GGML_METAL_LOG_INFO("%s: loading '%s'\n", __func__, [path_source UTF8String]);

            NSString * src = [NSString stringWithContentsOfFile:path_source encoding:NSUTF8StringEncoding error:&error];
            if (error) {
                GGML_METAL_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                return NULL;
            }
#endif // GGML_METAL_EMBED_LIBRARY

            @autoreleasepool {
                // dictionary of preprocessor macros
                NSMutableDictionary * prep = [NSMutableDictionary dictionary];

#ifdef GGML_QKK_64
                prep[@"GGML_QKK_64"] = @(1);
#endif

                MTLCompileOptions* options = [MTLCompileOptions new];
                options.preprocessorMacros = prep;

                //[options setFastMathEnabled:false];

                metal_library = [ctx->device newLibraryWithSource:src options:options error:&error];
                if (error) {
                    GGML_METAL_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                    return NULL;
                }
            }
        }
    }

    // print MTL GPU family:
    GGML_METAL_LOG_INFO("%s: GPU name:   %s\n", __func__, [[ctx->device name] UTF8String]);

    const NSInteger MTLGPUFamilyMetal3 = 5001;

    // determine max supported GPU family
    // https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
    // https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
    {
        for (int i = MTLGPUFamilyApple1 + 20; i >= MTLGPUFamilyApple1; --i) {
            if ([ctx->device supportsFamily:i]) {
                GGML_METAL_LOG_INFO("%s: GPU family: MTLGPUFamilyApple%d  (%d)\n", __func__, i - (int) MTLGPUFamilyApple1 + 1, i);
                break;
            }
        }

        for (int i = MTLGPUFamilyCommon1 + 5; i >= MTLGPUFamilyCommon1; --i) {
            if ([ctx->device supportsFamily:i]) {
                GGML_METAL_LOG_INFO("%s: GPU family: MTLGPUFamilyCommon%d (%d)\n", __func__, i - (int) MTLGPUFamilyCommon1 + 1, i);
                break;
            }
        }

        for (int i = MTLGPUFamilyMetal3 + 5; i >= MTLGPUFamilyMetal3; --i) {
            if ([ctx->device supportsFamily:i]) {
                GGML_METAL_LOG_INFO("%s: GPU family: MTLGPUFamilyMetal%d  (%d)\n", __func__, i - (int) MTLGPUFamilyMetal3 + 3, i);
                break;
            }
        }
    }

    ctx->support_simdgroup_reduction  = [ctx->device supportsFamily:MTLGPUFamilyApple7];
    ctx->support_simdgroup_reduction |= [ctx->device supportsFamily:MTLGPUFamilyMetal3];

    ctx->support_simdgroup_mm = [ctx->device supportsFamily:MTLGPUFamilyApple7];

    GGML_METAL_LOG_INFO("%s: simdgroup reduction support   = %s\n",       __func__, ctx->support_simdgroup_reduction ? "true" : "false");
    GGML_METAL_LOG_INFO("%s: simdgroup matrix mul. support = %s\n",       __func__, ctx->support_simdgroup_mm ? "true" : "false");
    GGML_METAL_LOG_INFO("%s: hasUnifiedMemory              = %s\n",       __func__, ctx->device.hasUnifiedMemory ? "true" : "false");

    ctx->should_capture_next_compute = false;

#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
    if (@available(macOS 10.12, iOS 16.0, *)) {
        GGML_METAL_LOG_INFO("%s: recommendedMaxWorkingSetSize  = %8.2f MB\n", __func__, ctx->device.recommendedMaxWorkingSetSize / 1e6);
    }
#elif TARGET_OS_OSX
    if (ctx->device.maxTransferRate != 0) {
        GGML_METAL_LOG_INFO("%s: maxTransferRate               = %8.2f MB/s\n", __func__, ctx->device.maxTransferRate / 1e6);
    } else {
        GGML_METAL_LOG_INFO("%s: maxTransferRate               = built-in GPU\n", __func__);
    }
#endif

    // load kernels
    {
        NSError * error = nil;

        for (int i = 0; i < GGML_METAL_KERNEL_TYPE_COUNT; ++i) {
            ctx->kernels[i].pipeline = nil;
        }

        /*
            GGML_METAL_LOG_INFO("%s: loaded %-32s %16p | th_max = %4d | th_width = %4d\n", __func__, "kernel_"#name, (void *) kernel->pipeline, \
                    (int) kernel->pipeline.maxTotalThreadsPerThreadgroup, \
                    (int) kernel->pipeline.threadExecutionWidth); \
        */
#define GGML_METAL_ADD_KERNEL(e, name, supported) \
        if (supported) { \
            struct ggml_metal_kernel * kernel = &ctx->kernels[e]; \
            id<MTLFunction> metal_function = [metal_library newFunctionWithName:@"kernel_"#name]; \
            kernel->pipeline = [ctx->device newComputePipelineStateWithFunction:metal_function error:&error]; \
            [metal_function release]; \
            if (error) { \
                GGML_METAL_LOG_ERROR("%s: error: load pipeline error: %s\n", __func__, [[error description] UTF8String]); \
                [metal_library release]; \
                return NULL; \
            } \
        } else { \
            GGML_METAL_LOG_WARN("%s: skipping %-32s (not supported)\n", __func__, "kernel_"#name); \
        }

        // simd_sum and simd_max requires MTLGPUFamilyApple7

        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_ADD,                       add,                    true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_ADD_ROW,                   add_row,                true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL,                       mul,                    true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_ROW,                   mul_row,                true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_DIV,                       div,                    true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_DIV_ROW,                   div_row,                true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_SCALE,                     scale,                  true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_SCALE_4,                   scale_4,                true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_CLAMP,                     clamp,                  true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_TANH,                      tanh,                   true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_RELU,                      relu,                   true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GELU,                      gelu,                   true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GELU_4,                    gelu_4,                 true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GELU_QUICK,                gelu_quick,             true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GELU_QUICK_4,              gelu_quick_4,           true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_SILU,                      silu,                   true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_SILU_4,                    silu_4,                 true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_SOFT_MAX,                  soft_max,               ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_SOFT_MAX_4,                soft_max_4,             ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_DIAG_MASK_INF,             diag_mask_inf,          true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_DIAG_MASK_INF_8,           diag_mask_inf_8,        true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GET_ROWS_F32,              get_rows_f32,           true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GET_ROWS_F16,              get_rows_f16,           true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GET_ROWS_Q4_0,             get_rows_q4_0,          true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GET_ROWS_Q4_1,             get_rows_q4_1,          true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GET_ROWS_Q5_0,             get_rows_q5_0,          true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GET_ROWS_Q5_1,             get_rows_q5_1,          true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GET_ROWS_Q8_0,             get_rows_q8_0,          true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GET_ROWS_Q2_K,             get_rows_q2_K,          true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GET_ROWS_Q3_K,             get_rows_q3_K,          true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GET_ROWS_Q4_K,             get_rows_q4_K,          true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GET_ROWS_Q5_K,             get_rows_q5_K,          true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GET_ROWS_Q6_K,             get_rows_q6_K,          true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ2_XXS,          get_rows_iq2_xxs,       true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ2_XS,           get_rows_iq2_xs,        true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ3_XXS,          get_rows_iq3_xxs,       true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ3_S,            get_rows_iq3_s,         true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ2_S,            get_rows_iq2_s,         true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ1_S,            get_rows_iq1_s,         true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ1_M,            get_rows_iq1_m,         true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ4_NL,           get_rows_iq4_nl,        true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ4_XS,           get_rows_iq4_xs,        true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GET_ROWS_I32,              get_rows_i32,           true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_RMS_NORM,                  rms_norm,               ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_GROUP_NORM,                group_norm,             ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_NORM,                      norm,                   true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_F32_F32,            mul_mv_f32_f32,         ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F16,            mul_mv_f16_f16,         ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F32,            mul_mv_f16_f32,         ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F32_1ROW,       mul_mv_f16_f32_1row,    ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F32_L4,         mul_mv_f16_f32_l4,      ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_Q4_0_F32,           mul_mv_q4_0_f32,        ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_Q4_1_F32,           mul_mv_q4_1_f32,        ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_Q5_0_F32,           mul_mv_q5_0_f32,        ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_Q5_1_F32,           mul_mv_q5_1_f32,        ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_Q8_0_F32,           mul_mv_q8_0_f32,        ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_Q2_K_F32,           mul_mv_q2_K_f32,        ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_Q3_K_F32,           mul_mv_q3_K_f32,        ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_Q4_K_F32,           mul_mv_q4_K_f32,        ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_Q5_K_F32,           mul_mv_q5_K_f32,        ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_Q6_K_F32,           mul_mv_q6_K_f32,        ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_IQ2_XXS_F32,        mul_mv_iq2_xxs_f32,     ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_IQ2_XS_F32,         mul_mv_iq2_xs_f32,      ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_IQ3_XXS_F32,        mul_mv_iq3_xxs_f32,     ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_IQ3_S_F32,          mul_mv_iq3_s_f32,       ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_IQ2_S_F32,          mul_mv_iq2_s_f32,       ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_IQ1_S_F32,          mul_mv_iq1_s_f32,       ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_IQ1_M_F32,          mul_mv_iq1_m_f32,       ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_IQ4_NL_F32,         mul_mv_iq4_nl_f32,      ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_IQ4_XS_F32,         mul_mv_iq4_xs_f32,      ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F32_F32,         mul_mv_id_f32_f32,      ctx->support_simdgroup_reduction);
      //GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F16_F16,         mul_mv_id_f16_f16,      ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F16_F32,         mul_mv_id_f16_f32,      ctx->support_simdgroup_reduction);
      //GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F16_F32_1ROW,    mul_mv_id_f16_f32_1row, ctx->support_simdgroup_reduction);
      //GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F16_F32_L4,      mul_mv_id_f16_f32_l4,   ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q4_0_F32,        mul_mv_id_q4_0_f32,     ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q4_1_F32,        mul_mv_id_q4_1_f32,     ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q5_0_F32,        mul_mv_id_q5_0_f32,     ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q5_1_F32,        mul_mv_id_q5_1_f32,     ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q8_0_F32,        mul_mv_id_q8_0_f32,     ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q2_K_F32,        mul_mv_id_q2_K_f32,     ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q3_K_F32,        mul_mv_id_q3_K_f32,     ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q4_K_F32,        mul_mv_id_q4_K_f32,     ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q5_K_F32,        mul_mv_id_q5_K_f32,     ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q6_K_F32,        mul_mv_id_q6_K_f32,     ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ2_XXS_F32,     mul_mv_id_iq2_xxs_f32,  ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ2_XS_F32,      mul_mv_id_iq2_xs_f32,   ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ3_XXS_F32,     mul_mv_id_iq3_xxs_f32,  ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ3_S_F32,       mul_mv_id_iq3_s_f32,    ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ2_S_F32,       mul_mv_id_iq2_s_f32,    ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ1_S_F32,       mul_mv_id_iq1_s_f32,    ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ1_M_F32,       mul_mv_id_iq1_m_f32,    ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ4_NL_F32,      mul_mv_id_iq4_nl_f32,   ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ4_XS_F32,      mul_mv_id_iq4_xs_f32,   ctx->support_simdgroup_reduction);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_F32_F32,            mul_mm_f32_f32,         ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_F16_F32,            mul_mm_f16_f32,         ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_Q4_0_F32,           mul_mm_q4_0_f32,        ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_Q4_1_F32,           mul_mm_q4_1_f32,        ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_Q5_0_F32,           mul_mm_q5_0_f32,        ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_Q5_1_F32,           mul_mm_q5_1_f32,        ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_Q8_0_F32,           mul_mm_q8_0_f32,        ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_Q2_K_F32,           mul_mm_q2_K_f32,        ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_Q3_K_F32,           mul_mm_q3_K_f32,        ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_Q4_K_F32,           mul_mm_q4_K_f32,        ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_Q5_K_F32,           mul_mm_q5_K_f32,        ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_Q6_K_F32,           mul_mm_q6_K_f32,        ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_IQ2_XXS_F32,        mul_mm_iq2_xxs_f32,     ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_IQ2_XS_F32,         mul_mm_iq2_xs_f32,      ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_IQ3_XXS_F32,        mul_mm_iq3_xxs_f32,     ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_IQ3_S_F32,          mul_mm_iq3_s_f32,       ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_IQ2_S_F32,          mul_mm_iq2_s_f32,       ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_IQ1_S_F32,          mul_mm_iq1_s_f32,       ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_IQ1_M_F32,          mul_mm_iq1_m_f32,       ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_IQ4_NL_F32,         mul_mm_iq4_nl_f32,      ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_IQ4_XS_F32,         mul_mm_iq4_xs_f32,      ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_ID_F32_F32,         mul_mm_id_f32_f32,      ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_ID_F16_F32,         mul_mm_id_f16_f32,      ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q4_0_F32,        mul_mm_id_q4_0_f32,     ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q4_1_F32,        mul_mm_id_q4_1_f32,     ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q5_0_F32,        mul_mm_id_q5_0_f32,     ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q5_1_F32,        mul_mm_id_q5_1_f32,     ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q8_0_F32,        mul_mm_id_q8_0_f32,     ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q2_K_F32,        mul_mm_id_q2_K_f32,     ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q3_K_F32,        mul_mm_id_q3_K_f32,     ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q4_K_F32,        mul_mm_id_q4_K_f32,     ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q5_K_F32,        mul_mm_id_q5_K_f32,     ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q6_K_F32,        mul_mm_id_q6_K_f32,     ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ2_XXS_F32,     mul_mm_id_iq2_xxs_f32,  ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ2_XS_F32,      mul_mm_id_iq2_xs_f32,   ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ3_XXS_F32,     mul_mm_id_iq3_xxs_f32,  ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ3_S_F32,       mul_mm_id_iq3_s_f32,    ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ2_S_F32,       mul_mm_id_iq2_s_f32,    ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ1_S_F32,       mul_mm_id_iq1_s_f32,    ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ1_M_F32,       mul_mm_id_iq1_m_f32,    ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ4_NL_F32,      mul_mm_id_iq4_nl_f32,   ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ4_XS_F32,      mul_mm_id_iq4_xs_f32,   ctx->support_simdgroup_mm);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_ROPE_F32,                  rope_f32,               true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_ROPE_F16,                  rope_f16,               true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_ALIBI_F32,                 alibi_f32,              true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_IM2COL_F16,                im2col_f16,             true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_IM2COL_F32,                im2col_f32,             true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_UPSCALE_F32,               upscale_f32,            true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_PAD_F32,                   pad_f32,                true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_TIMESTEP_EMBEDDING_F32,    timestep_embedding_f32, true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_ARANGE_F32,                arange_f32,             true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_ARGSORT_F32_I32_ASC,       argsort_f32_i32_asc,    true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_ARGSORT_F32_I32_DESC,      argsort_f32_i32_desc,   true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_LEAKY_RELU_F32,            leaky_relu_f32,         true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_CPY_F32_F16,               cpy_f32_f16,            true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_CPY_F32_F32,               cpy_f32_f32,            true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_CPY_F32_Q8_0,              cpy_f32_q8_0,           true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_CPY_F32_Q4_0,              cpy_f32_q4_0,           true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_CPY_F32_Q4_1,              cpy_f32_q4_1,           true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_CPY_F32_Q5_0,              cpy_f32_q5_0,           true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_CPY_F32_Q5_1,              cpy_f32_q5_1,           true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_CPY_F32_IQ4_NL,            cpy_f32_iq4_nl,         true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_CPY_F16_F16,               cpy_f16_f16,            true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_CPY_F16_F32,               cpy_f16_f32,            true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_CONCAT,                    concat,                 true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_SQR,                       sqr,                    true);
        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_SUM_ROWS,                  sum_rows,               true);
    }

    [metal_library release];
    return ctx;
}

static void ggml_metal_free(struct ggml_metal_context * ctx) {
    GGML_METAL_LOG_INFO("%s: deallocating\n", __func__);

    for (int i = 0; i < GGML_METAL_KERNEL_TYPE_COUNT; ++i) {
        [ctx->kernels[i].pipeline release];
    }

    [ctx->queue release];
    [ctx->device release];

    dispatch_release(ctx->d_queue);

    free(ctx);
}

// temporarily defined here for compatibility between ggml-backend and the old API

struct ggml_backend_metal_buffer {
    void   * data;
    size_t   size;

    id<MTLBuffer> metal;
};

struct ggml_backend_metal_buffer_context {
    void * all_data;
    size_t all_size;
    bool owned;

    // multiple buffers are used only to avoid the maximum buffer size limitation when using mmap
    int n_buffers;
    struct ggml_backend_metal_buffer buffers[GGML_METAL_MAX_BUFFERS];
};

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
static id<MTLBuffer> ggml_metal_get_buffer(struct ggml_tensor * t, size_t * offs) {
    //GGML_METAL_LOG_INFO("%s: data tensor '%16s', offs_data = %8ld, offs_eval = %8ld, offs_cach = %8ld\n", __func__, t->name, offs_data, offs_eval, offs_cach);

    const int64_t tsize = ggml_nbytes(t);

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    struct ggml_backend_metal_buffer_context * buf_ctx = (struct ggml_backend_metal_buffer_context *) buffer->context;

    // find the view that contains the tensor fully
    for (int i = 0; i < buf_ctx->n_buffers; ++i) {
        const int64_t ioffs = (int64_t) t->data - (int64_t) buf_ctx->buffers[i].data;

        //GGML_METAL_LOG_INFO("ioffs = %10ld, tsize = %10ld, sum = %10ld, buf_ctx->buffers[%d].size = %10ld\n", ioffs, tsize, ioffs + tsize, i, buf_ctx->buffers[i].size);
        if (ioffs >= 0 && ioffs + tsize <= (int64_t) buf_ctx->buffers[i].size) {
            *offs = (size_t) ioffs;

            //GGML_METAL_LOG_INFO("%s: tensor '%16s', offs = %8ld\n", __func__, t->name, *offs);

            return buf_ctx->buffers[i].metal;
        }
    }

    GGML_METAL_LOG_ERROR("%s: error: tensor '%s' buffer is nil\n", __func__, t->name);

    return nil;
}

static bool ggml_metal_supports_op(const struct ggml_metal_context * ctx, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                    return true;
                default:
                    return false;
            }
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
        case GGML_OP_CONCAT:
        case GGML_OP_ADD:
        case GGML_OP_ACC:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SCALE:
        case GGML_OP_CLAMP:
        case GGML_OP_SQR:
        case GGML_OP_SUM_ROWS:
            return true;
        case GGML_OP_SOFT_MAX:
        case GGML_OP_RMS_NORM:
        case GGML_OP_GROUP_NORM:
            return ctx->support_simdgroup_reduction;
        case GGML_OP_NORM:
        case GGML_OP_ALIBI:
        case GGML_OP_ROPE:
        case GGML_OP_IM2COL:
            return true;
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
            return false;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_ARGSORT:
        case GGML_OP_LEAKY_RELU:
            return true;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            return ctx->support_simdgroup_reduction &&
                (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F32);
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        switch (op->type) {
                           case GGML_TYPE_F16:
                           case GGML_TYPE_F32:
                           case GGML_TYPE_Q8_0:
                           case GGML_TYPE_Q4_0:
                           case GGML_TYPE_Q4_1:
                           case GGML_TYPE_Q5_0:
                           case GGML_TYPE_Q5_1:
                           case GGML_TYPE_IQ4_NL:
                                return true;
                           default:
                                return false;
                        }
                    case GGML_TYPE_F16:
                        switch (op->type) {
                           case GGML_TYPE_F16:
                           case GGML_TYPE_F32:
                                return true;
                           default:
                                return false;
                        }
                    default:
                        return false;
                };
            }
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_GET_ROWS:
            {
                return op->ne[3] == 1;
            }
        default:
            return false;
    }
}

static enum ggml_status ggml_metal_graph_compute(
        struct ggml_metal_context * ctx,
               struct ggml_cgraph * gf) {

    @autoreleasepool {
    MTLComputePassDescriptor * edesc = MTLComputePassDescriptor.computePassDescriptor;
    edesc.dispatchType = MTLDispatchTypeSerial;

    // create multiple command buffers and enqueue them
    // then, we encode the graph into the command buffers in parallel

    const int n_nodes  = gf->n_nodes;
    const int n_cb = ctx->n_cb;
    const int n_nodes_per_cb = (n_nodes + n_cb - 1) / n_cb;

    const bool should_capture = ctx->should_capture_next_compute;
    if (should_capture) {
        ctx->should_capture_next_compute = false;

        MTLCaptureDescriptor * descriptor = [MTLCaptureDescriptor new];
        descriptor.captureObject = ctx->queue;

        NSError * error = nil;
        if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error]) {
            GGML_METAL_LOG_ERROR("%s: error: unable to start capture '%s'\n", __func__, [[error localizedDescription] UTF8String]);
            GGML_ASSERT(!"capture failed");
        }
    }

    id<MTLCommandBuffer> command_buffer_builder[n_cb];
    for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
        id<MTLCommandBuffer> command_buffer  = [ctx->queue commandBufferWithUnretainedReferences];
        command_buffer_builder[cb_idx] = command_buffer;

        // enqueue the command buffers in order to specify their execution order
        [command_buffer enqueue];
    }

    const id<MTLCommandBuffer> *command_buffers = command_buffer_builder;

    dispatch_apply(n_cb, ctx->d_queue, ^(size_t iter) {
        const int cb_idx = iter;

        size_t offs_src0 = 0;
        size_t offs_src1 = 0;
        size_t offs_src2 = 0;
        size_t offs_dst  = 0;

        id<MTLCommandBuffer> command_buffer  = command_buffers[cb_idx];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];

        const int node_start =                                      (cb_idx + 0) * n_nodes_per_cb;
        const int node_end   = MIN((cb_idx == n_cb - 1) ? n_nodes : (cb_idx + 1) * n_nodes_per_cb, n_nodes);

        for (int i = node_start; i < node_end; ++i) {
            if (i == -1) {
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                continue;
            }

            //GGML_METAL_LOG_INFO("%s: encoding node %3d, op = %8s\n", __func__, i, ggml_op_name(gf->nodes[i]->op));

            struct ggml_tensor * src0 = gf->nodes[i]->src[0];
            struct ggml_tensor * src1 = gf->nodes[i]->src[1];
            struct ggml_tensor * src2 = gf->nodes[i]->src[2];
            struct ggml_tensor * dst  = gf->nodes[i];

            if (ggml_is_empty(dst)) {
                continue;
            }

            switch (dst->op) {
                case GGML_OP_NONE:
                case GGML_OP_RESHAPE:
                case GGML_OP_VIEW:
                case GGML_OP_TRANSPOSE:
                case GGML_OP_PERMUTE:
                    {
                        // noop -> next node
                    } continue;
                default:
                    {
                    } break;
            }

            if (!ggml_metal_supports_op(ctx, dst)) {
                GGML_METAL_LOG_ERROR("%s: error: unsupported op '%s'\n", __func__, ggml_op_desc(dst));
                GGML_ASSERT(!"unsupported op");
            }

            if (should_capture) {
                [encoder pushDebugGroup:[NSString stringWithCString:ggml_op_desc(dst) encoding:NSUTF8StringEncoding]];
            }

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

            id<MTLBuffer> id_src0 = src0 ? ggml_metal_get_buffer(src0, &offs_src0) : nil;
            id<MTLBuffer> id_src1 = src1 ? ggml_metal_get_buffer(src1, &offs_src1) : nil;
            id<MTLBuffer> id_src2 = src2 ? ggml_metal_get_buffer(src2, &offs_src2) : nil;
            id<MTLBuffer> id_dst  = dst  ? ggml_metal_get_buffer(dst,  &offs_dst)  : nil;

            //GGML_METAL_LOG_INFO("%s: op - %s\n", __func__, ggml_op_name(dst->op));
            //if (src0) {
            //    GGML_METAL_LOG_INFO("%s: src0 - %4s [%5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(src0t), ne00, ne01, ne02,
            //            ggml_is_contiguous(src0), src0->name);
            //}
            //if (src1) {
            //    GGML_METAL_LOG_INFO("%s: src1 - %4s [%5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(src1t), ne10, ne11, ne12,
            //            ggml_is_contiguous(src1), src1->name);
            //}
            //if (dst) {
            //    GGML_METAL_LOG_INFO("%s: dst  - %4s [%5lld, %5lld, %5lld], 1, %s\n",  __func__, ggml_type_name(dstt),  ne0,  ne1,  ne2,
            //            dst->name);
            //}

            switch (dst->op) {
                case GGML_OP_CONCAT:
                    {
                        const int64_t nb = ne00;

                        id<MTLComputePipelineState> pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_CONCAT].pipeline;

                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                        [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                        [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                        [encoder setBytes:&ne00 length:sizeof(ne00) atIndex:3];
                        [encoder setBytes:&ne01 length:sizeof(ne01) atIndex:4];
                        [encoder setBytes:&ne02 length:sizeof(ne02) atIndex:5];
                        [encoder setBytes:&ne03 length:sizeof(ne03) atIndex:6];
                        [encoder setBytes:&nb00 length:sizeof(nb00) atIndex:7];
                        [encoder setBytes:&nb01 length:sizeof(nb01) atIndex:8];
                        [encoder setBytes:&nb02 length:sizeof(nb02) atIndex:9];
                        [encoder setBytes:&nb03 length:sizeof(nb03) atIndex:10];
                        [encoder setBytes:&ne10 length:sizeof(ne10) atIndex:11];
                        [encoder setBytes:&ne11 length:sizeof(ne11) atIndex:12];
                        [encoder setBytes:&ne12 length:sizeof(ne12) atIndex:13];
                        [encoder setBytes:&ne13 length:sizeof(ne13) atIndex:14];
                        [encoder setBytes:&nb10 length:sizeof(nb10) atIndex:15];
                        [encoder setBytes:&nb11 length:sizeof(nb11) atIndex:16];
                        [encoder setBytes:&nb12 length:sizeof(nb12) atIndex:17];
                        [encoder setBytes:&nb13 length:sizeof(nb13) atIndex:18];
                        [encoder setBytes:&ne0  length:sizeof(ne0)  atIndex:19];
                        [encoder setBytes:&ne1  length:sizeof(ne1)  atIndex:20];
                        [encoder setBytes:&ne2  length:sizeof(ne2)  atIndex:21];
                        [encoder setBytes:&ne3  length:sizeof(ne3)  atIndex:22];
                        [encoder setBytes:&nb0  length:sizeof(nb0)  atIndex:23];
                        [encoder setBytes:&nb1  length:sizeof(nb1)  atIndex:24];
                        [encoder setBytes:&nb2  length:sizeof(nb2)  atIndex:25];
                        [encoder setBytes:&nb3  length:sizeof(nb3)  atIndex:26];
                        [encoder setBytes:&nb   length:sizeof(nb)   atIndex:27];

                        const int nth = MIN(1024, ne0);

                        [encoder dispatchThreadgroups:MTLSizeMake(ne1, ne2, ne3) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                    } break;
                case GGML_OP_ADD:
                case GGML_OP_MUL:
                case GGML_OP_DIV:
                    {
                        const size_t offs = 0;

                        bool bcast_row = false;

                        int64_t nb = ne00;

                        id<MTLComputePipelineState> pipeline = nil;

                        if (ggml_nelements(src1) == ne10 && ggml_is_contiguous(src1) && ne00 % 4 == 0 && ne10 % 4 == 0) {
                            GGML_ASSERT(ggml_is_contiguous(src0));

                            // src1 is a row
                            GGML_ASSERT(ne11 == 1);

                            nb = ne00 / 4;
                            switch (dst->op) {
                                case GGML_OP_ADD: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_ADD_ROW].pipeline; break;
                                case GGML_OP_MUL: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_ROW].pipeline; break;
                                case GGML_OP_DIV: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_DIV_ROW].pipeline; break;
                                default: GGML_ASSERT(false);
                            }

                            bcast_row = true;
                        } else {
                            switch (dst->op) {
                                case GGML_OP_ADD: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_ADD].pipeline; break;
                                case GGML_OP_MUL: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL].pipeline; break;
                                case GGML_OP_DIV: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_DIV].pipeline; break;
                                default: GGML_ASSERT(false);
                            }
                        }

                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                        [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                        [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                        [encoder setBytes:&ne00 length:sizeof(ne00) atIndex:3];
                        [encoder setBytes:&ne01 length:sizeof(ne01) atIndex:4];
                        [encoder setBytes:&ne02 length:sizeof(ne02) atIndex:5];
                        [encoder setBytes:&ne03 length:sizeof(ne03) atIndex:6];
                        [encoder setBytes:&nb00 length:sizeof(nb00) atIndex:7];
                        [encoder setBytes:&nb01 length:sizeof(nb01) atIndex:8];
                        [encoder setBytes:&nb02 length:sizeof(nb02) atIndex:9];
                        [encoder setBytes:&nb03 length:sizeof(nb03) atIndex:10];
                        [encoder setBytes:&ne10 length:sizeof(ne10) atIndex:11];
                        [encoder setBytes:&ne11 length:sizeof(ne11) atIndex:12];
                        [encoder setBytes:&ne12 length:sizeof(ne12) atIndex:13];
                        [encoder setBytes:&ne13 length:sizeof(ne13) atIndex:14];
                        [encoder setBytes:&nb10 length:sizeof(nb10) atIndex:15];
                        [encoder setBytes:&nb11 length:sizeof(nb11) atIndex:16];
                        [encoder setBytes:&nb12 length:sizeof(nb12) atIndex:17];
                        [encoder setBytes:&nb13 length:sizeof(nb13) atIndex:18];
                        [encoder setBytes:&ne0  length:sizeof(ne0)  atIndex:19];
                        [encoder setBytes:&ne1  length:sizeof(ne1)  atIndex:20];
                        [encoder setBytes:&ne2  length:sizeof(ne2)  atIndex:21];
                        [encoder setBytes:&ne3  length:sizeof(ne3)  atIndex:22];
                        [encoder setBytes:&nb0  length:sizeof(nb0)  atIndex:23];
                        [encoder setBytes:&nb1  length:sizeof(nb1)  atIndex:24];
                        [encoder setBytes:&nb2  length:sizeof(nb2)  atIndex:25];
                        [encoder setBytes:&nb3  length:sizeof(nb3)  atIndex:26];
                        [encoder setBytes:&offs length:sizeof(offs) atIndex:27];
                        [encoder setBytes:&nb   length:sizeof(nb)   atIndex:28];

                        if (bcast_row) {
                            const int64_t n = ggml_nelements(dst)/4;

                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } else {
                            const int nth = MIN((int) pipeline.maxTotalThreadsPerThreadgroup, ne0);

                            [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        }
                    } break;
                case GGML_OP_ACC:
                    {
                        GGML_ASSERT(src0t == GGML_TYPE_F32);
                        GGML_ASSERT(src1t == GGML_TYPE_F32);
                        GGML_ASSERT(dstt  == GGML_TYPE_F32);

                        GGML_ASSERT(ggml_is_contiguous(src0));
                        GGML_ASSERT(ggml_is_contiguous(src1));

                        const size_t pnb1 = ((int32_t *) dst->op_params)[0];
                        const size_t pnb2 = ((int32_t *) dst->op_params)[1];
                        const size_t pnb3 = ((int32_t *) dst->op_params)[2];
                        const size_t offs = ((int32_t *) dst->op_params)[3];

                        const bool inplace = (bool) ((int32_t *) dst->op_params)[4];

                        if (!inplace) {
                            // run a separete kernel to cpy src->dst
                            // not sure how to avoid this
                            // TODO: make a simpler cpy_bytes kernel

                            const id<MTLComputePipelineState> pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_CPY_F32_F32].pipeline;

                            [encoder setComputePipelineState:pipeline];
                            [encoder setBuffer:id_src0 offset:offs_src0        atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst         atIndex:1];
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

                            const int nth = MIN((int) pipeline.maxTotalThreadsPerThreadgroup, ne00);

                            [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        }

                        const id<MTLComputePipelineState> pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_ADD].pipeline;

                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                        [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                        [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                        [encoder setBytes:&ne00 length:sizeof(ne00) atIndex:3];
                        [encoder setBytes:&ne01 length:sizeof(ne01) atIndex:4];
                        [encoder setBytes:&ne02 length:sizeof(ne02) atIndex:5];
                        [encoder setBytes:&ne03 length:sizeof(ne03) atIndex:6];
                        [encoder setBytes:&nb00 length:sizeof(nb00) atIndex:7];
                        [encoder setBytes:&pnb1 length:sizeof(pnb1) atIndex:8];
                        [encoder setBytes:&pnb2 length:sizeof(pnb2) atIndex:9];
                        [encoder setBytes:&pnb3 length:sizeof(pnb3) atIndex:10];
                        [encoder setBytes:&ne10 length:sizeof(ne10) atIndex:11];
                        [encoder setBytes:&ne11 length:sizeof(ne11) atIndex:12];
                        [encoder setBytes:&ne12 length:sizeof(ne12) atIndex:13];
                        [encoder setBytes:&ne13 length:sizeof(ne13) atIndex:14];
                        [encoder setBytes:&nb10 length:sizeof(nb10) atIndex:15];
                        [encoder setBytes:&nb11 length:sizeof(nb11) atIndex:16];
                        [encoder setBytes:&nb12 length:sizeof(nb12) atIndex:17];
                        [encoder setBytes:&nb13 length:sizeof(nb13) atIndex:18];
                        [encoder setBytes:&ne0  length:sizeof(ne0)  atIndex:19];
                        [encoder setBytes:&ne1  length:sizeof(ne1)  atIndex:20];
                        [encoder setBytes:&ne2  length:sizeof(ne2)  atIndex:21];
                        [encoder setBytes:&ne3  length:sizeof(ne3)  atIndex:22];
                        [encoder setBytes:&nb0  length:sizeof(nb0)  atIndex:23];
                        [encoder setBytes:&pnb1 length:sizeof(pnb1) atIndex:24];
                        [encoder setBytes:&pnb2 length:sizeof(pnb2) atIndex:25];
                        [encoder setBytes:&pnb3 length:sizeof(pnb3) atIndex:26];
                        [encoder setBytes:&offs length:sizeof(offs) atIndex:27];

                        const int nth = MIN((int) pipeline.maxTotalThreadsPerThreadgroup, ne00);

                        [encoder dispatchThreadgroups:MTLSizeMake(ne11, ne12, ne13) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                    } break;
                case GGML_OP_SCALE:
                    {
                        GGML_ASSERT(ggml_is_contiguous(src0));

                        float scale;
                        memcpy(&scale, dst->op_params, sizeof(scale));

                        int64_t n = ggml_nelements(dst);

                        id<MTLComputePipelineState> pipeline = nil;

                        if (n % 4 == 0) {
                            n /= 4;
                            pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_SCALE_4].pipeline;
                        } else {
                            pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_SCALE].pipeline;
                        }

                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:id_src0   offset:offs_src0 atIndex:0];
                        [encoder setBuffer:id_dst    offset:offs_dst  atIndex:1];
                        [encoder setBytes:&scale length:sizeof(scale) atIndex:2];

                        [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                    } break;
                case GGML_OP_CLAMP:
                {
                    id<MTLComputePipelineState> pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_CLAMP].pipeline;

                    float min;
                    float max;
                    memcpy(&min, ((int32_t *) dst->op_params) + 0, sizeof(float));
                    memcpy(&max, ((int32_t *) dst->op_params) + 1, sizeof(float));

                    [encoder setComputePipelineState:pipeline];
                    [encoder setBuffer:id_src0   offset:offs_src0 atIndex:0];
                    [encoder setBuffer:id_dst    offset:offs_dst  atIndex:1];
                    [encoder setBytes:&min length:sizeof(min) atIndex:2];
                    [encoder setBytes:&max length:sizeof(max) atIndex:3];

                    const int64_t n = ggml_nelements(dst);

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } break;
                case GGML_OP_UNARY:
                    switch (ggml_get_unary_op(gf->nodes[i])) {
                        // we are not taking into account the strides, so for now require contiguous tensors
                        GGML_ASSERT(ggml_is_contiguous(src0));

                        case GGML_UNARY_OP_TANH:
                            {
                                id<MTLComputePipelineState> pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_TANH].pipeline;

                                [encoder setComputePipelineState:pipeline];
                                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                                const int64_t n = ggml_nelements(dst);

                                [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                            } break;
                        case GGML_UNARY_OP_RELU:
                            {
                                id<MTLComputePipelineState> pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_RELU].pipeline;

                                [encoder setComputePipelineState:pipeline];
                                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                                const int64_t n = ggml_nelements(dst);

                                [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                            } break;
                        case GGML_UNARY_OP_GELU:
                            {
                                int64_t n = ggml_nelements(dst);

                                id<MTLComputePipelineState> pipeline = nil;

                                if (n % 4 == 0) {
                                    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GELU_4].pipeline;
                                    n /= 4;
                                } else {
                                    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GELU].pipeline;
                                }

                                [encoder setComputePipelineState:pipeline];
                                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                                [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                            } break;
                        case GGML_UNARY_OP_GELU_QUICK:
                            {
                                int64_t n = ggml_nelements(dst);

                                id<MTLComputePipelineState> pipeline = nil;

                                if (n % 4 == 0) {
                                    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GELU_QUICK_4].pipeline;
                                    n /= 4;
                                } else {
                                    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GELU_QUICK].pipeline;
                                }

                                [encoder setComputePipelineState:pipeline];
                                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                                [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                            } break;
                        case GGML_UNARY_OP_SILU:
                            {
                                int64_t n = ggml_nelements(dst);

                                id<MTLComputePipelineState> pipeline = nil;

                                if (n % 4 == 0) {
                                    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_SILU_4].pipeline;
                                    n /= 4;
                                } else {
                                    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_SILU].pipeline;
                                }

                                [encoder setComputePipelineState:pipeline];
                                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                                [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                            } break;
                        default:
                            {
                                GGML_METAL_LOG_WARN("%s: node %3d, op = %8s not implemented\n", __func__, i, ggml_op_name(dst->op));
                                GGML_ASSERT(false);
                            }
                    } break;
                case GGML_OP_SQR:
                    {
                        GGML_ASSERT(ggml_is_contiguous(src0));

                        id<MTLComputePipelineState> pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_SQR].pipeline;

                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                        [encoder setBuffer:id_dst  offset:offs_dst atIndex:1];

                        const int64_t n = ggml_nelements(dst);

                        [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                    } break;
                case GGML_OP_SUM_ROWS:
                    {
                        GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));

                        id<MTLComputePipelineState> pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_SUM_ROWS].pipeline;

                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                        [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                        [encoder setBytes:&ne00 length:sizeof(ne00) atIndex:2];
                        [encoder setBytes:&ne01 length:sizeof(ne01) atIndex:3];
                        [encoder setBytes:&ne02 length:sizeof(ne02) atIndex:4];
                        [encoder setBytes:&ne03 length:sizeof(ne03) atIndex:5];
                        [encoder setBytes:&nb00 length:sizeof(nb00) atIndex:6];
                        [encoder setBytes:&nb01 length:sizeof(nb01) atIndex:7];
                        [encoder setBytes:&nb02 length:sizeof(nb02) atIndex:8];
                        [encoder setBytes:&nb03 length:sizeof(nb03) atIndex:9];
                        [encoder setBytes:&ne10 length:sizeof(ne10) atIndex:10];
                        [encoder setBytes:&ne11 length:sizeof(ne11) atIndex:11];
                        [encoder setBytes:&ne12 length:sizeof(ne12) atIndex:12];
                        [encoder setBytes:&ne13 length:sizeof(ne13) atIndex:13];
                        [encoder setBytes:&nb10 length:sizeof(nb10) atIndex:14];
                        [encoder setBytes:&nb11 length:sizeof(nb11) atIndex:15];
                        [encoder setBytes:&nb12 length:sizeof(nb12) atIndex:16];
                        [encoder setBytes:&nb13 length:sizeof(nb13) atIndex:17];
                        [encoder setBytes:&ne0  length:sizeof(ne0)  atIndex:18];
                        [encoder setBytes:&ne1  length:sizeof(ne1)  atIndex:19];
                        [encoder setBytes:&ne2  length:sizeof(ne2)  atIndex:20];
                        [encoder setBytes:&ne3  length:sizeof(ne3)  atIndex:21];
                        [encoder setBytes:&nb0  length:sizeof(nb0)  atIndex:22];
                        [encoder setBytes:&nb1  length:sizeof(nb1)  atIndex:23];
                        [encoder setBytes:&nb2  length:sizeof(nb2)  atIndex:24];
                        [encoder setBytes:&nb3  length:sizeof(nb3)  atIndex:25];

                        [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                    } break;
                case GGML_OP_SOFT_MAX:
                    {
                        int nth = 32; // SIMD width

                        id<MTLComputePipelineState> pipeline = nil;

                        if (ne00%4 == 0) {
                            while (nth < ne00/4 && nth < 256) {
                                nth *= 2;
                            }
                            pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_SOFT_MAX_4].pipeline;
                        } else {
                            while (nth < ne00 && nth < 1024) {
                                nth *= 2;
                            }
                            pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_SOFT_MAX].pipeline;
                        }

                        float scale;
                        float max_bias;

                        memcpy(&scale,    ((int32_t *) dst->op_params) + 0, sizeof(scale));
                        memcpy(&max_bias, ((int32_t *) dst->op_params) + 1, sizeof(max_bias));

                        const int64_t nrows_x = ggml_nrows(src0);
                        const int64_t nrows_y = src0->ne[1];

                        const uint32_t n_head_kv   = nrows_x/nrows_y;
                        const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head_kv));

                        const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
                        const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:id_src0 offset:offs_src0   atIndex:0];
                        if (id_src1) {
                            [encoder setBuffer:id_src1 offset:offs_src1   atIndex:1];
                        } else {
                            [encoder setBuffer:id_src0 offset:offs_src0   atIndex:1];
                        }
                        if (id_src2) {
                            [encoder setBuffer:id_src2 offset:offs_src2   atIndex:2];
                        } else {
                            [encoder setBuffer:id_src0 offset:offs_src0   atIndex:2];
                        }
                        [encoder setBuffer:id_dst   offset:offs_dst          atIndex:3];
                        [encoder setBytes:&ne00     length:sizeof(ne00)      atIndex:4];
                        [encoder setBytes:&ne01     length:sizeof(ne01)      atIndex:5];
                        [encoder setBytes:&ne02     length:sizeof(ne02)      atIndex:6];
                        [encoder setBytes:&scale    length:sizeof(scale)     atIndex:7];
                        [encoder setBytes:&max_bias length:sizeof(max_bias)  atIndex:8];
                        [encoder setBytes:&m0       length:sizeof(m0)        atIndex:9];
                        [encoder setBytes:&m1       length:sizeof(m1)        atIndex:10];
                        [encoder setBytes:&n_head_log2 length:sizeof(n_head_log2) atIndex:11];
                        [encoder setThreadgroupMemoryLength:32*sizeof(float) atIndex:0];

                        [encoder dispatchThreadgroups:MTLSizeMake(ne01*ne02*ne03, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                    } break;
                case GGML_OP_DIAG_MASK_INF:
                    {
                        const int n_past = ((int32_t *)(dst->op_params))[0];

                        id<MTLComputePipelineState> pipeline = nil;

                        if (ne00%8 == 0) {
                            pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_DIAG_MASK_INF_8].pipeline;
                        } else {
                            pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_DIAG_MASK_INF].pipeline;
                        }

                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                        [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                        [encoder setBytes:&ne00   length:sizeof(ne00) atIndex:2];
                        [encoder setBytes:&ne01   length:sizeof(ne01) atIndex:3];
                        [encoder setBytes:&n_past length:sizeof(int)  atIndex:4];

                        if (ne00%8 == 0) {
                            [encoder dispatchThreadgroups:MTLSizeMake(ne00*ne01*ne02/8, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        }
                        else {
                            [encoder dispatchThreadgroups:MTLSizeMake(ne00, ne01, ne02) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        }
                    } break;
                case GGML_OP_MUL_MAT:
                    {
                        GGML_ASSERT(ne00 == ne10);

                        // TODO: assert that dim2 and dim3 are contiguous
                        GGML_ASSERT(ne12 % ne02 == 0);
                        GGML_ASSERT(ne13 % ne03 == 0);

                        const uint r2 = ne12/ne02;
                        const uint r3 = ne13/ne03;

                        // find the break-even point where the matrix-matrix kernel becomes more efficient compared
                        // to the matrix-vector kernel
                        int ne11_mm_min = 1;

#if 0
                        // the numbers below are measured on M2 Ultra for 7B and 13B models
                        // these numbers do not translate to other devices or model sizes
                        // TODO: need to find a better approach
                        if ([ctx->device.name isEqualToString:@"Apple M2 Ultra"]) {
                            switch (src0t) {
                                case GGML_TYPE_F16:  ne11_mm_min = 2;  break;
                                case GGML_TYPE_Q8_0: ne11_mm_min = 7;  break;
                                case GGML_TYPE_Q2_K: ne11_mm_min = 15; break;
                                case GGML_TYPE_Q3_K: ne11_mm_min = 7;  break;
                                case GGML_TYPE_Q4_0:
                                case GGML_TYPE_Q4_1: ne11_mm_min = 15; break;
                                case GGML_TYPE_Q4_K: ne11_mm_min = 11; break;
                                case GGML_TYPE_Q5_0:                          // not tested yet
                                case GGML_TYPE_Q5_1: ne11_mm_min = 13; break; // not tested yet
                                case GGML_TYPE_Q5_K: ne11_mm_min = 7;  break;
                                case GGML_TYPE_Q6_K: ne11_mm_min = 7;  break;
                                default:             ne11_mm_min = 1;  break;
                            }
                        }
#endif

                        // for now the matrix-matrix multiplication kernel only works on A14+/M1+ SoCs
                        // AMD GPU and older A-chips will reuse matrix-vector multiplication kernel
                        if ([ctx->device supportsFamily:MTLGPUFamilyApple7] &&
                            !ggml_is_transposed(src0) &&
                            !ggml_is_transposed(src1) &&
                            src1t == GGML_TYPE_F32 &&
                            ne00 % 32 == 0 && ne00 >= 64 &&
                            (ne11 > ne11_mm_min || (ggml_is_quantized(src0t) && ne12 > 1))) {
                            //printf("matrix: ne00 = %6d, ne01 = %6d, ne02 = %6d, ne11 = %6d, ne12 = %6d\n", ne00, ne01, ne02, ne11, ne12);

                            // some Metal matrix data types require aligned pointers
                            // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
                            switch (src0->type) {
                                case GGML_TYPE_F32: GGML_ASSERT(nb01 % 16 == 0); break;
                                case GGML_TYPE_F16: GGML_ASSERT(nb01 % 8  == 0); break;
                                default: break;
                            }

                            id<MTLComputePipelineState> pipeline = nil;

                            switch (src0->type) {
                                case GGML_TYPE_F32:     pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_F32_F32    ].pipeline; break;
                                case GGML_TYPE_F16:     pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_F16_F32    ].pipeline; break;
                                case GGML_TYPE_Q4_0:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_Q4_0_F32   ].pipeline; break;
                                case GGML_TYPE_Q4_1:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_Q4_1_F32   ].pipeline; break;
                                case GGML_TYPE_Q5_0:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_Q5_0_F32   ].pipeline; break;
                                case GGML_TYPE_Q5_1:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_Q5_1_F32   ].pipeline; break;
                                case GGML_TYPE_Q8_0:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_Q8_0_F32   ].pipeline; break;
                                case GGML_TYPE_Q2_K:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_Q2_K_F32   ].pipeline; break;
                                case GGML_TYPE_Q3_K:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_Q3_K_F32   ].pipeline; break;
                                case GGML_TYPE_Q4_K:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_Q4_K_F32   ].pipeline; break;
                                case GGML_TYPE_Q5_K:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_Q5_K_F32   ].pipeline; break;
                                case GGML_TYPE_Q6_K:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_Q6_K_F32   ].pipeline; break;
                                case GGML_TYPE_IQ2_XXS: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_IQ2_XXS_F32].pipeline; break;
                                case GGML_TYPE_IQ2_XS:  pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_IQ2_XS_F32 ].pipeline; break;
                                case GGML_TYPE_IQ3_XXS: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_IQ3_XXS_F32].pipeline; break;
                                case GGML_TYPE_IQ3_S:   pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_IQ3_S_F32  ].pipeline; break;
                                case GGML_TYPE_IQ2_S:   pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_IQ2_S_F32  ].pipeline; break;
                                case GGML_TYPE_IQ1_S:   pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_IQ1_S_F32  ].pipeline; break;
                                case GGML_TYPE_IQ1_M:   pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_IQ1_M_F32  ].pipeline; break;
                                case GGML_TYPE_IQ4_NL:  pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_IQ4_NL_F32 ].pipeline; break;
                                case GGML_TYPE_IQ4_XS:  pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_IQ4_XS_F32 ].pipeline; break;
                                default: GGML_ASSERT(false && "MUL MAT-MAT not implemented");
                            }

                            [encoder setComputePipelineState:pipeline];
                            [encoder setBuffer:id_src0 offset:offs_src0    atIndex:0];
                            [encoder setBuffer:id_src1 offset:offs_src1    atIndex:1];
                            [encoder setBuffer:id_dst  offset:offs_dst     atIndex:2];
                            [encoder setBytes:&ne00    length:sizeof(ne00) atIndex:3];
                            [encoder setBytes:&ne02    length:sizeof(ne02) atIndex:4];
                            [encoder setBytes:&nb01    length:sizeof(nb01) atIndex:5];
                            [encoder setBytes:&nb02    length:sizeof(nb02) atIndex:6];
                            [encoder setBytes:&ne12    length:sizeof(ne12) atIndex:7];
                            [encoder setBytes:&nb10    length:sizeof(nb10) atIndex:8];
                            [encoder setBytes:&nb11    length:sizeof(nb11) atIndex:9];
                            [encoder setBytes:&nb12    length:sizeof(nb12) atIndex:10];
                            [encoder setBytes:&ne0     length:sizeof(ne0)  atIndex:11];
                            [encoder setBytes:&ne1     length:sizeof(ne1)  atIndex:12];
                            [encoder setBytes:&r2      length:sizeof(r2)   atIndex:13];
                            [encoder setBytes:&r3      length:sizeof(r3)   atIndex:14];
                            [encoder setThreadgroupMemoryLength:8192 atIndex:0];
                            [encoder dispatchThreadgroups:MTLSizeMake( (ne11 + 31)/32, (ne01 + 63)/64, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        } else {
                            int nth0 = 32;
                            int nth1 = 1;
                            int nrows = 1;
                            //printf("vector: ne00 = %6d, ne01 = %6d, ne02 = %6d, ne11 = %6d, ne12 = %6d\n", ne00, ne01, ne02, ne11, ne12);

                            id<MTLComputePipelineState> pipeline = nil;

                            // use custom matrix x vector kernel
                            switch (src0t) {
                                case GGML_TYPE_F32:
                                    {
                                        GGML_ASSERT(src1t == GGML_TYPE_F32);
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_F32_F32].pipeline;
                                        nrows = 4;
                                    } break;
                                case GGML_TYPE_F16:
                                    {
                                        nth0 = 32;
                                        nth1 = 1;
                                        if (src1t == GGML_TYPE_F32) {
                                            if (ne11 * ne12 < 4) {
                                                pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F32_1ROW].pipeline;
                                            } else if (ne00 >= 128 && ne01 >= 8 && ne00%4 == 0) {
                                                pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F32_L4].pipeline;
                                                nrows = ne11;
                                            } else {
                                                pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F32].pipeline;
                                                nrows = 4;
                                            }
                                        } else {
                                            pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F16].pipeline;
                                            nrows = 4;
                                        }
                                    } break;
                                case GGML_TYPE_Q4_0:
                                    {
                                        nth0 = 8;
                                        nth1 = 8;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_Q4_0_F32].pipeline;
                                    } break;
                                case GGML_TYPE_Q4_1:
                                    {
                                        nth0 = 8;
                                        nth1 = 8;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_Q4_1_F32].pipeline;
                                    } break;
                                case GGML_TYPE_Q5_0:
                                    {
                                        nth0 = 8;
                                        nth1 = 8;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_Q5_0_F32].pipeline;
                                    } break;
                                case GGML_TYPE_Q5_1:
                                    {
                                        nth0 = 8;
                                        nth1 = 8;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_Q5_1_F32].pipeline;
                                    } break;
                                case GGML_TYPE_Q8_0:
                                    {
                                        nth0 = 8;
                                        nth1 = 8;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_Q8_0_F32].pipeline;
                                    } break;
                                case GGML_TYPE_Q2_K:
                                    {
                                        nth0 = 2;
                                        nth1 = 32;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_Q2_K_F32].pipeline;
                                    } break;
                                case GGML_TYPE_Q3_K:
                                    {
                                        nth0 = 2;
                                        nth1 = 32;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_Q3_K_F32].pipeline;
                                    } break;
                                case GGML_TYPE_Q4_K:
                                    {
                                        nth0 = 4; //1;
                                        nth1 = 8; //32;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_Q4_K_F32].pipeline;
                                    } break;
                                case GGML_TYPE_Q5_K:
                                    {
                                        nth0 = 2;
                                        nth1 = 32;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_Q5_K_F32].pipeline;
                                    } break;
                                case GGML_TYPE_Q6_K:
                                    {
                                        nth0 = 2;
                                        nth1 = 32;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_Q6_K_F32].pipeline;
                                    } break;
                                case GGML_TYPE_IQ2_XXS:
                                    {
                                        nth0 = 4;
                                        nth1 = 16;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_IQ2_XXS_F32].pipeline;
                                    } break;
                                case GGML_TYPE_IQ2_XS:
                                    {
                                        nth0 = 4;
                                        nth1 = 16;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_IQ2_XS_F32].pipeline;
                                    } break;
                                case GGML_TYPE_IQ3_XXS:
                                    {
                                        nth0 = 4;
                                        nth1 = 16;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_IQ3_XXS_F32].pipeline;
                                    } break;
                                case GGML_TYPE_IQ3_S:
                                    {
                                        nth0 = 4;
                                        nth1 = 16;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_IQ3_S_F32].pipeline;
                                    } break;
                                case GGML_TYPE_IQ2_S:
                                    {
                                        nth0 = 4;
                                        nth1 = 16;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_IQ2_S_F32].pipeline;
                                    } break;
                                case GGML_TYPE_IQ1_S:
                                    {
                                        nth0 = 4;
                                        nth1 = 16;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_IQ1_S_F32].pipeline;
                                    } break;
                                case GGML_TYPE_IQ1_M:
                                    {
                                        nth0 = 4;
                                        nth1 = 16;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_IQ1_M_F32].pipeline;
                                    } break;
                                case GGML_TYPE_IQ4_NL:
                                    {
                                        nth0 = 4;
                                        nth1 = 16;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_IQ4_NL_F32].pipeline;
                                    } break;
                                case GGML_TYPE_IQ4_XS:
                                    {
                                        nth0 = 4;
                                        nth1 = 16;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_IQ4_XS_F32].pipeline;
                                    } break;
                                default:
                                    {
                                        GGML_METAL_LOG_ERROR("Asserting on type %d\n", (int)src0t);
                                        GGML_ASSERT(false && "not implemented");
                                    }
                            };

                            if (ggml_is_quantized(src0t)) {
                                GGML_ASSERT(ne00 >= nth0*nth1);
                            }

                            [encoder setComputePipelineState:pipeline];
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
                            [encoder setBytes:&r2   length:sizeof(r2)   atIndex:17];
                            [encoder setBytes:&r3   length:sizeof(r3)   atIndex:18];

                            if (src0t == GGML_TYPE_Q4_0  || src0t == GGML_TYPE_Q4_1  || src0t == GGML_TYPE_Q5_0 ||
                                src0t == GGML_TYPE_Q5_1  || src0t == GGML_TYPE_Q8_0  || src0t == GGML_TYPE_Q2_K ||
                                src0t == GGML_TYPE_IQ1_S || src0t == GGML_TYPE_IQ1_M || src0t == GGML_TYPE_IQ2_S) {
                                [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 7)/8, ne11, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                            }
                            else if (src0t == GGML_TYPE_IQ2_XXS || src0t == GGML_TYPE_IQ2_XS) {
                                const int mem_size = src0t == GGML_TYPE_IQ2_XXS ? 256*8+128 : 512*8+128;
                                [encoder setThreadgroupMemoryLength:mem_size atIndex:0];
                                [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 7)/8, ne11, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                            }
                            else if (src0t == GGML_TYPE_IQ3_XXS || src0t == GGML_TYPE_IQ3_S) {
                                const int mem_size = src0t == GGML_TYPE_IQ3_XXS ? 256*4+128 : 512*4;
                                [encoder setThreadgroupMemoryLength:mem_size atIndex:0];
                                [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 7)/8, ne11, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                            }
                            else if (src0t == GGML_TYPE_IQ4_NL || src0t == GGML_TYPE_IQ4_XS) {
                                const int mem_size = 32*sizeof(float);
                                [encoder setThreadgroupMemoryLength:mem_size atIndex:0];
                                [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 3)/4, ne11, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                            }
                            else if (src0t == GGML_TYPE_Q4_K) {
                                [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 3)/4, ne11, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                            }
                            else if (src0t == GGML_TYPE_Q3_K) {
#ifdef GGML_QKK_64
                                [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 1)/2, ne11, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
#else
                                [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 3)/4, ne11, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
#endif
                            }
                            else if (src0t == GGML_TYPE_Q5_K) {
                                [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 3)/4, ne11, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                            }
                            else if (src0t == GGML_TYPE_Q6_K) {
                                [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 1)/2, ne11, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                            } else {
                                const int64_t ny = (ne11 + nrows - 1)/nrows;
                                [encoder dispatchThreadgroups:MTLSizeMake(ne01, ny, ne12*ne13) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                            }
                        }
                    } break;
                case GGML_OP_MUL_MAT_ID:
                    {
                        const int n_as = src0->ne[2];

                        // src2 = ids
                        const int64_t  ne20 = src2->ne[0];
                        const int64_t  ne21 = src2->ne[1];
                        const int64_t  ne22 = src2->ne[2]; GGML_UNUSED(ne22);
                        const int64_t  ne23 = src2->ne[3]; GGML_UNUSED(ne23);

                        const uint64_t nb20 = src2->nb[0]; GGML_UNUSED(nb20);
                        const uint64_t nb21 = src2->nb[1];
                        const uint64_t nb22 = src2->nb[2]; GGML_UNUSED(nb22);
                        const uint64_t nb23 = src2->nb[3]; GGML_UNUSED(nb23);

                        const enum ggml_type src2t = src2->type; GGML_UNUSED(src2t);

                        GGML_ASSERT(src2t == GGML_TYPE_I32);

                        GGML_ASSERT(!ggml_is_transposed(src0));
                        GGML_ASSERT(!ggml_is_transposed(src1));

                        GGML_ASSERT(src1t == GGML_TYPE_F32);

                        // find the break-even point where the matrix-matrix kernel becomes more efficient compared
                        // to the matrix-vector kernel
                        // ne20 = n_used_experts
                        // ne21 = n_rows
                        const int dst_rows = ne20*ne21;
                        const int dst_rows_min = n_as;

                        // max size of the rowids array in the kernel shared buffer
                        GGML_ASSERT(dst_rows <= 2048);

                        // for now the matrix-matrix multiplication kernel only works on A14+/M1+ SoCs
                        // AMD GPU and older A-chips will reuse matrix-vector multiplication kernel
                        // !!!
                        // TODO: for now, always use mat-vec kernels until we figure out how to improve the
                        //       indirect matrix multiplication
                        // !!!
                        if ([ctx->device supportsFamily:MTLGPUFamilyApple7] &&
                            ne00 % 32 == 0 && ne00 >= 64 &&
                            dst_rows > dst_rows_min) {

                            // some Metal matrix data types require aligned pointers
                            // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
                            switch (src0->type) {
                                case GGML_TYPE_F32: GGML_ASSERT(nb01 % 16 == 0); break;
                                case GGML_TYPE_F16: GGML_ASSERT(nb01 % 8  == 0); break;
                                default: break;
                            }

                            id<MTLComputePipelineState> pipeline = nil;

                            switch (src0->type) {
                                case GGML_TYPE_F32:     pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_ID_F32_F32    ].pipeline; break;
                                case GGML_TYPE_F16:     pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_ID_F16_F32    ].pipeline; break;
                                case GGML_TYPE_Q4_0:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q4_0_F32   ].pipeline; break;
                                case GGML_TYPE_Q4_1:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q4_1_F32   ].pipeline; break;
                                case GGML_TYPE_Q5_0:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q5_0_F32   ].pipeline; break;
                                case GGML_TYPE_Q5_1:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q5_1_F32   ].pipeline; break;
                                case GGML_TYPE_Q8_0:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q8_0_F32   ].pipeline; break;
                                case GGML_TYPE_Q2_K:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q2_K_F32   ].pipeline; break;
                                case GGML_TYPE_Q3_K:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q3_K_F32   ].pipeline; break;
                                case GGML_TYPE_Q4_K:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q4_K_F32   ].pipeline; break;
                                case GGML_TYPE_Q5_K:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q5_K_F32   ].pipeline; break;
                                case GGML_TYPE_Q6_K:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q6_K_F32   ].pipeline; break;
                                case GGML_TYPE_IQ2_XXS: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ2_XXS_F32].pipeline; break;
                                case GGML_TYPE_IQ2_XS:  pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ2_XS_F32 ].pipeline; break;
                                case GGML_TYPE_IQ3_XXS: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ3_XXS_F32].pipeline; break;
                                case GGML_TYPE_IQ3_S:   pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ3_S_F32  ].pipeline; break;
                                case GGML_TYPE_IQ2_S:   pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ2_S_F32  ].pipeline; break;
                                case GGML_TYPE_IQ1_S:   pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ1_S_F32  ].pipeline; break;
                                case GGML_TYPE_IQ1_M:   pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ1_M_F32  ].pipeline; break;
                                case GGML_TYPE_IQ4_NL:  pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ4_NL_F32 ].pipeline; break;
                                case GGML_TYPE_IQ4_XS:  pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ4_XS_F32 ].pipeline; break;
                                default: GGML_ASSERT(false && "MUL_MAT_ID not implemented");
                            }

                            [encoder setComputePipelineState:pipeline];
                            [encoder setBuffer:id_src0 offset:offs_src0    atIndex:0];
                            [encoder setBuffer:id_src1 offset:offs_src1    atIndex:1];
                            [encoder setBuffer:id_dst  offset:offs_dst     atIndex:2];
                            [encoder setBuffer:id_src2 offset:offs_src2    atIndex:3];
                            [encoder setBytes:&ne20    length:sizeof(ne20) atIndex:4];
                            [encoder setBytes:&ne21    length:sizeof(ne21) atIndex:5];
                            [encoder setBytes:&nb21    length:sizeof(nb21) atIndex:6];
                            [encoder setBytes:&ne00    length:sizeof(ne00) atIndex:7];
                            [encoder setBytes:&ne02    length:sizeof(ne02) atIndex:8];
                            [encoder setBytes:&nb01    length:sizeof(nb01) atIndex:9];
                            [encoder setBytes:&nb02    length:sizeof(nb02) atIndex:10];
                            [encoder setBytes:&ne11    length:sizeof(ne11) atIndex:11];
                            [encoder setBytes:&ne12    length:sizeof(ne12) atIndex:12];
                            [encoder setBytes:&ne13    length:sizeof(ne13) atIndex:13];
                            [encoder setBytes:&nb10    length:sizeof(nb10) atIndex:14];
                            [encoder setBytes:&nb11    length:sizeof(nb11) atIndex:15];
                            [encoder setBytes:&nb12    length:sizeof(nb12) atIndex:16];
                            [encoder setBytes:&ne0     length:sizeof(ne0)  atIndex:17];
                            [encoder setBytes:&ne1     length:sizeof(ne1)  atIndex:18];
                            [encoder setBytes:&nb1     length:sizeof(nb1)  atIndex:19];

                            [encoder setThreadgroupMemoryLength:GGML_PAD(8192 + dst_rows*4/*sizeof(ushort2)*/, 16) atIndex:0];

                            [encoder dispatchThreadgroups:MTLSizeMake((ne21 + 31)/32, (ne01 + 63)/64, n_as) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        } else {
                            int nth0 = 32;
                            int nth1 = 1;
                            int nrows = 1;
                            //printf("vector: ne00 = %6d, ne01 = %6d, ne02 = %6d, ne11 = %6d, ne12 = %6d\n", ne00, ne01, ne02, ne11, ne12);

                            id<MTLComputePipelineState> pipeline = nil;

                            // use custom matrix x vector kernel
                            switch (src0t) {
                                case GGML_TYPE_F32:
                                    {
                                        GGML_ASSERT(src1t == GGML_TYPE_F32);
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F32_F32].pipeline;
                                    } break;
                                case GGML_TYPE_F16:
                                    {
                                        GGML_ASSERT(src1t == GGML_TYPE_F32);
                                        nth0 = 32;
                                        nth1 = 1;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F16_F32].pipeline;
                                    } break;
                                case GGML_TYPE_Q4_0:
                                    {
                                        nth0 = 8;
                                        nth1 = 8;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q4_0_F32].pipeline;
                                    } break;
                                case GGML_TYPE_Q4_1:
                                    {
                                        nth0 = 8;
                                        nth1 = 8;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q4_1_F32].pipeline;
                                    } break;
                                case GGML_TYPE_Q5_0:
                                    {
                                        nth0 = 8;
                                        nth1 = 8;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q5_0_F32].pipeline;
                                    } break;
                                case GGML_TYPE_Q5_1:
                                    {
                                        nth0 = 8;
                                        nth1 = 8;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q5_1_F32].pipeline;
                                    } break;
                                case GGML_TYPE_Q8_0:
                                    {
                                        nth0 = 8;
                                        nth1 = 8;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q8_0_F32].pipeline;
                                    } break;
                                case GGML_TYPE_Q2_K:
                                    {
                                        nth0 = 2;
                                        nth1 = 32;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q2_K_F32].pipeline;
                                    } break;
                                case GGML_TYPE_Q3_K:
                                    {
                                        nth0 = 2;
                                        nth1 = 32;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q3_K_F32].pipeline;
                                    } break;
                                case GGML_TYPE_Q4_K:
                                    {
                                        nth0 = 4; //1;
                                        nth1 = 8; //32;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q4_K_F32].pipeline;
                                    } break;
                                case GGML_TYPE_Q5_K:
                                    {
                                        nth0 = 2;
                                        nth1 = 32;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q5_K_F32].pipeline;
                                    } break;
                                case GGML_TYPE_Q6_K:
                                    {
                                        nth0 = 2;
                                        nth1 = 32;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q6_K_F32].pipeline;
                                    } break;
                                case GGML_TYPE_IQ2_XXS:
                                    {
                                        nth0 = 4;
                                        nth1 = 16;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ2_XXS_F32].pipeline;
                                    } break;
                                case GGML_TYPE_IQ2_XS:
                                    {
                                        nth0 = 4;
                                        nth1 = 16;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ2_XS_F32].pipeline;
                                    } break;
                                case GGML_TYPE_IQ3_XXS:
                                    {
                                        nth0 = 4;
                                        nth1 = 16;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ3_XXS_F32].pipeline;
                                    } break;
                                case GGML_TYPE_IQ3_S:
                                    {
                                        nth0 = 4;
                                        nth1 = 16;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ3_S_F32].pipeline;
                                    } break;
                                case GGML_TYPE_IQ2_S:
                                    {
                                        nth0 = 4;
                                        nth1 = 16;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ2_S_F32].pipeline;
                                    } break;
                                case GGML_TYPE_IQ1_S:
                                    {
                                        nth0 = 4;
                                        nth1 = 16;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ1_S_F32].pipeline;
                                    } break;
                                case GGML_TYPE_IQ1_M:
                                    {
                                        nth0 = 4;
                                        nth1 = 16;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ1_M_F32].pipeline;
                                    } break;
                                case GGML_TYPE_IQ4_NL:
                                    {
                                        nth0 = 4;
                                        nth1 = 16;
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ4_NL_F32].pipeline;
                                    } break;
                                case GGML_TYPE_IQ4_XS:
                                    {
                                        nth0 = 4;
                                        nth1 = 16;
                                    #if QK_K == 64
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ4_NL_F32].pipeline;
                                    #else
                                        pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ4_XS_F32].pipeline;
                                    #endif

                                    } break;
                                default:
                                    {
                                        GGML_METAL_LOG_ERROR("Asserting on type %d\n", (int)src2t);
                                        GGML_ASSERT(false && "not implemented");
                                    }
                            };

                            if (ggml_is_quantized(src0t)) {
                                GGML_ASSERT(ne00 >= nth0*nth1);
                            }

                            [encoder setComputePipelineState:pipeline];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                            [encoder setBuffer:id_src2 offset:offs_src2 atIndex:3];
                            [encoder setBytes:&ne20 length:sizeof(ne20) atIndex:4];
                            [encoder setBytes:&ne21 length:sizeof(ne21) atIndex:5];
                            [encoder setBytes:&nb21 length:sizeof(nb21) atIndex:6];
                            [encoder setBytes:&ne00 length:sizeof(ne00) atIndex:7];
                            [encoder setBytes:&ne01 length:sizeof(ne01) atIndex:8];
                            [encoder setBytes:&ne02 length:sizeof(ne02) atIndex:9];
                            [encoder setBytes:&nb00 length:sizeof(nb00) atIndex:10];
                            [encoder setBytes:&nb01 length:sizeof(nb01) atIndex:11];
                            [encoder setBytes:&nb02 length:sizeof(nb02) atIndex:12];
                            [encoder setBytes:&ne10 length:sizeof(ne10) atIndex:13];
                            [encoder setBytes:&ne11 length:sizeof(ne11) atIndex:14];
                            [encoder setBytes:&ne12 length:sizeof(ne12) atIndex:15];
                            [encoder setBytes:&ne13 length:sizeof(ne13) atIndex:16];
                            [encoder setBytes:&nb10 length:sizeof(nb10) atIndex:17];
                            [encoder setBytes:&nb11 length:sizeof(nb11) atIndex:18];
                            [encoder setBytes:&nb12 length:sizeof(nb12) atIndex:19];
                            [encoder setBytes:&ne0  length:sizeof(ne0)  atIndex:20];
                            [encoder setBytes:&ne1  length:sizeof(ne1)  atIndex:21];
                            [encoder setBytes:&nb1  length:sizeof(nb1)  atIndex:22];

                            const int64_t _ne1 = 1;
                            const int tgz = dst_rows;

                            if (src0t == GGML_TYPE_Q4_0  || src0t == GGML_TYPE_Q4_1  || src0t == GGML_TYPE_Q5_0 ||
                                src0t == GGML_TYPE_Q5_1  || src0t == GGML_TYPE_Q8_0  || src0t == GGML_TYPE_Q2_K ||
                                src0t == GGML_TYPE_IQ1_S || src0t == GGML_TYPE_IQ1_M || src0t == GGML_TYPE_IQ2_S) {
                                [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 7)/8, _ne1, tgz) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                            }
                            else if (src0t == GGML_TYPE_IQ2_XXS || src0t == GGML_TYPE_IQ2_XS) {
                                const int mem_size = src0t == GGML_TYPE_IQ2_XXS ? 256*8+128 : 512*8+128;
                                [encoder setThreadgroupMemoryLength:mem_size atIndex:0];
                                [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 7)/8, _ne1, tgz) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                            }
                            else if (src0t == GGML_TYPE_IQ3_XXS || src0t == GGML_TYPE_IQ3_S) {
                                const int mem_size = src0t == GGML_TYPE_IQ3_XXS ? 256*4+128 : 512*4;
                                [encoder setThreadgroupMemoryLength:mem_size atIndex:0];
                                [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 7)/8, _ne1, tgz) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                            }
                            else if (src0t == GGML_TYPE_IQ4_NL || src0t == GGML_TYPE_IQ4_XS) {
                                const int mem_size = 32*sizeof(float);
                                [encoder setThreadgroupMemoryLength:mem_size atIndex:0];
                                [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 3)/4, _ne1, tgz) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                            }
                            else if (src0t == GGML_TYPE_Q4_K) {
                                [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 3)/4, _ne1, tgz) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                            }
                            else if (src0t == GGML_TYPE_Q3_K) {
#ifdef GGML_QKK_64
                                [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 1)/2, _ne1, tgz) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
#else
                                [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 3)/4, _ne1, tgz) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
#endif
                            }
                            else if (src0t == GGML_TYPE_Q5_K) {
                                [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 3)/4, _ne1, tgz) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                            }
                            else if (src0t == GGML_TYPE_Q6_K) {
                                [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 1)/2, _ne1, tgz) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                            } else {
                                const int64_t ny = (_ne1 + nrows - 1)/nrows; // = _ne1
                                [encoder dispatchThreadgroups:MTLSizeMake(ne01, ny, tgz) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                            }
                        }
                    } break;
                case GGML_OP_GET_ROWS:
                    {
                        id<MTLComputePipelineState> pipeline = nil;

                        switch (src0->type) {
                            case GGML_TYPE_F32:     pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GET_ROWS_F32    ].pipeline; break;
                            case GGML_TYPE_F16:     pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GET_ROWS_F16    ].pipeline; break;
                            case GGML_TYPE_Q4_0:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GET_ROWS_Q4_0   ].pipeline; break;
                            case GGML_TYPE_Q4_1:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GET_ROWS_Q4_1   ].pipeline; break;
                            case GGML_TYPE_Q5_0:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GET_ROWS_Q5_0   ].pipeline; break;
                            case GGML_TYPE_Q5_1:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GET_ROWS_Q5_1   ].pipeline; break;
                            case GGML_TYPE_Q8_0:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GET_ROWS_Q8_0   ].pipeline; break;
                            case GGML_TYPE_Q2_K:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GET_ROWS_Q2_K   ].pipeline; break;
                            case GGML_TYPE_Q3_K:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GET_ROWS_Q3_K   ].pipeline; break;
                            case GGML_TYPE_Q4_K:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GET_ROWS_Q4_K   ].pipeline; break;
                            case GGML_TYPE_Q5_K:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GET_ROWS_Q5_K   ].pipeline; break;
                            case GGML_TYPE_Q6_K:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GET_ROWS_Q6_K   ].pipeline; break;
                            case GGML_TYPE_IQ2_XXS: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ2_XXS].pipeline; break;
                            case GGML_TYPE_IQ2_XS:  pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ2_XS ].pipeline; break;
                            case GGML_TYPE_IQ3_XXS: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ3_XXS].pipeline; break;
                            case GGML_TYPE_IQ3_S:   pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ3_S  ].pipeline; break;
                            case GGML_TYPE_IQ2_S:   pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ2_S  ].pipeline; break;
                            case GGML_TYPE_IQ1_S:   pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ1_S  ].pipeline; break;
                            case GGML_TYPE_IQ1_M:   pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ1_M  ].pipeline; break;
                            case GGML_TYPE_IQ4_NL:  pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ4_NL ].pipeline; break;
                            case GGML_TYPE_IQ4_XS:  pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ4_XS ].pipeline; break;
                            case GGML_TYPE_I32:     pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GET_ROWS_I32    ].pipeline; break;
                            default: GGML_ASSERT(false && "not implemented");
                        }

                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:id_src0     offset:offs_src0 atIndex:0];
                        [encoder setBuffer:id_src1     offset:offs_src1 atIndex:1];
                        [encoder setBuffer:id_dst      offset:offs_dst  atIndex:2];
                        [encoder setBytes:&ne00 length:sizeof( int64_t) atIndex:3];
                        [encoder setBytes:&nb01 length:sizeof(uint64_t) atIndex:4];
                        [encoder setBytes:&nb02 length:sizeof(uint64_t) atIndex:5];
                        [encoder setBytes:&ne10 length:sizeof( int64_t) atIndex:6];
                        [encoder setBytes:&nb10 length:sizeof( int64_t) atIndex:7];
                        [encoder setBytes:&nb11 length:sizeof( int64_t) atIndex:8];
                        [encoder setBytes:&nb1  length:sizeof(uint64_t) atIndex:9];
                        [encoder setBytes:&nb2  length:sizeof(uint64_t) atIndex:10];

                        [encoder dispatchThreadgroups:MTLSizeMake(ne10, ne11, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    } break;
                case GGML_OP_RMS_NORM:
                    {
                        GGML_ASSERT(ne00 % 4 == 0);

                        float eps;
                        memcpy(&eps, dst->op_params, sizeof(float));

                        int nth = 32; // SIMD width

                        while (nth < ne00/4 && nth < 1024) {
                            nth *= 2;
                        }

                        id<MTLComputePipelineState> pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_RMS_NORM].pipeline;

                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:id_src0 offset:offs_src0        atIndex:0];
                        [encoder setBuffer:id_dst  offset:offs_dst         atIndex:1];
                        [encoder setBytes:&ne00    length:sizeof( int64_t) atIndex:2];
                        [encoder setBytes:&nb01    length:sizeof(uint64_t) atIndex:3];
                        [encoder setBytes:&eps     length:sizeof(   float) atIndex:4];
                        [encoder setThreadgroupMemoryLength:32*sizeof(float) atIndex:0];

                        const int64_t nrows = ggml_nrows(src0);

                        [encoder dispatchThreadgroups:MTLSizeMake(nrows, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                    } break;
                case GGML_OP_GROUP_NORM:
                    {
                        GGML_ASSERT(ne00 % 4 == 0);

                        //float eps;
                        //memcpy(&eps, dst->op_params, sizeof(float));

                        const float eps = 1e-6f; // TODO: temporarily hardcoded

                        const int32_t n_groups = ((int32_t *) dst->op_params)[0];

                        int nth = 32; // SIMD width

                        //while (nth < ne00/4 && nth < 1024) {
                        //    nth *= 2;
                        //}

                        id<MTLComputePipelineState> pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_GROUP_NORM].pipeline;

                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:id_src0  offset:offs_src0        atIndex:0];
                        [encoder setBuffer:id_dst   offset:offs_dst         atIndex:1];
                        [encoder setBytes:&ne00     length:sizeof( int64_t) atIndex:2];
                        [encoder setBytes:&ne01     length:sizeof( int64_t) atIndex:3];
                        [encoder setBytes:&ne02     length:sizeof( int64_t) atIndex:4];
                        [encoder setBytes:&nb00     length:sizeof(uint64_t) atIndex:5];
                        [encoder setBytes:&nb01     length:sizeof(uint64_t) atIndex:6];
                        [encoder setBytes:&nb02     length:sizeof(uint64_t) atIndex:7];
                        [encoder setBytes:&n_groups length:sizeof( int32_t) atIndex:8];
                        [encoder setBytes:&eps      length:sizeof(   float) atIndex:9];
                        [encoder setThreadgroupMemoryLength:32*sizeof(float) atIndex:0];

                        [encoder dispatchThreadgroups:MTLSizeMake(n_groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                    } break;
                case GGML_OP_NORM:
                    {
                        float eps;
                        memcpy(&eps, dst->op_params, sizeof(float));

                        const int nth = MIN(256, ne00);

                        id<MTLComputePipelineState> pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_NORM].pipeline;

                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:id_src0 offset:offs_src0        atIndex:0];
                        [encoder setBuffer:id_dst  offset:offs_dst         atIndex:1];
                        [encoder setBytes:&ne00    length:sizeof( int64_t) atIndex:2];
                        [encoder setBytes:&nb01    length:sizeof(uint64_t) atIndex:3];
                        [encoder setBytes:&eps     length:sizeof(   float) atIndex:4];
                        [encoder setThreadgroupMemoryLength:GGML_PAD(nth*sizeof(float), 16) atIndex:0];

                        const int64_t nrows = ggml_nrows(src0);

                        [encoder dispatchThreadgroups:MTLSizeMake(nrows, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                    } break;
                case GGML_OP_ALIBI:
                    {
                        GGML_ASSERT((src0t == GGML_TYPE_F32));

                        const int nth = MIN(1024, ne00);

                        //const int n_past = ((int32_t *) dst->op_params)[0];
                        const int n_head = ((int32_t *) dst->op_params)[1];

                        float max_bias;
                        memcpy(&max_bias, (int32_t *) dst->op_params + 2, sizeof(float));

                        const int n_heads_log2_floor = 1 << (int) floor(log2(n_head));
                        const float m0 = powf(2.0f, -(max_bias) / n_heads_log2_floor);
                        const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_heads_log2_floor);

                        id<MTLComputePipelineState> pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_ALIBI_F32].pipeline;

                        [encoder setComputePipelineState:pipeline];
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
                        [encoder setBytes:&m0   length:sizeof(   float) atIndex:18];
                        [encoder setBytes:&m1   length:sizeof(   float) atIndex:19];
                        [encoder setBytes:&n_heads_log2_floor   length:sizeof(int) atIndex:20];

                        [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                    } break;
                case GGML_OP_ROPE:
                    {
                        GGML_ASSERT(ne10 == ne02);

                        const int nth = MIN(1024, ne00);

                        const int n_past     = ((int32_t *) dst->op_params)[0];
                        const int n_dims     = ((int32_t *) dst->op_params)[1];
                        const int mode       = ((int32_t *) dst->op_params)[2];
                        // skip 3, n_ctx, used in GLM RoPE, unimplemented in metal
                        const int n_orig_ctx = ((int32_t *) dst->op_params)[4];

                        float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
                        memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
                        memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
                        memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
                        memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
                        memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
                        memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));

                        id<MTLComputePipelineState> pipeline = nil;

                        switch (src0->type) {
                            case GGML_TYPE_F32: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_ROPE_F32].pipeline; break;
                            case GGML_TYPE_F16: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_ROPE_F16].pipeline; break;
                            default: GGML_ASSERT(false);
                        };

                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:id_src0     offset:offs_src0        atIndex:0];
                        [encoder setBuffer:id_src1     offset:offs_src1        atIndex:1];
                        [encoder setBuffer:id_dst      offset:offs_dst         atIndex:2];
                        [encoder setBytes:&ne00        length:sizeof( int64_t) atIndex:3];
                        [encoder setBytes:&ne01        length:sizeof( int64_t) atIndex:4];
                        [encoder setBytes:&ne02        length:sizeof( int64_t) atIndex:5];
                        [encoder setBytes:&ne03        length:sizeof( int64_t) atIndex:6];
                        [encoder setBytes:&nb00        length:sizeof(uint64_t) atIndex:7];
                        [encoder setBytes:&nb01        length:sizeof(uint64_t) atIndex:8];
                        [encoder setBytes:&nb02        length:sizeof(uint64_t) atIndex:9];
                        [encoder setBytes:&nb03        length:sizeof(uint64_t) atIndex:10];
                        [encoder setBytes:&ne0         length:sizeof( int64_t) atIndex:11];
                        [encoder setBytes:&ne1         length:sizeof( int64_t) atIndex:12];
                        [encoder setBytes:&ne2         length:sizeof( int64_t) atIndex:13];
                        [encoder setBytes:&ne3         length:sizeof( int64_t) atIndex:14];
                        [encoder setBytes:&nb0         length:sizeof(uint64_t) atIndex:15];
                        [encoder setBytes:&nb1         length:sizeof(uint64_t) atIndex:16];
                        [encoder setBytes:&nb2         length:sizeof(uint64_t) atIndex:17];
                        [encoder setBytes:&nb3         length:sizeof(uint64_t) atIndex:18];
                        [encoder setBytes:&n_past      length:sizeof(     int) atIndex:19];
                        [encoder setBytes:&n_dims      length:sizeof(     int) atIndex:20];
                        [encoder setBytes:&mode        length:sizeof(     int) atIndex:21];
                        [encoder setBytes:&n_orig_ctx  length:sizeof(     int) atIndex:22];
                        [encoder setBytes:&freq_base   length:sizeof(   float) atIndex:23];
                        [encoder setBytes:&freq_scale  length:sizeof(   float) atIndex:24];
                        [encoder setBytes:&ext_factor  length:sizeof(   float) atIndex:25];
                        [encoder setBytes:&attn_factor length:sizeof(   float) atIndex:26];
                        [encoder setBytes:&beta_fast   length:sizeof(   float) atIndex:27];
                        [encoder setBytes:&beta_slow   length:sizeof(   float) atIndex:28];

                        [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                    } break;
                case GGML_OP_IM2COL:
                    {
                        GGML_ASSERT(src0->type == GGML_TYPE_F16);
                        GGML_ASSERT(src1->type == GGML_TYPE_F32);
                        GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

                        const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
                        const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
                        const int32_t p0 = ((const int32_t *)(dst->op_params))[2];
                        const int32_t p1 = ((const int32_t *)(dst->op_params))[3];
                        const int32_t d0 = ((const int32_t *)(dst->op_params))[4];
                        const int32_t d1 = ((const int32_t *)(dst->op_params))[5];

                        const bool is_2D = ((const int32_t *)(dst->op_params))[6] == 1;

                        const int32_t N  = src1->ne[is_2D ? 3 : 2];
                        const int32_t IC = src1->ne[is_2D ? 2 : 1];
                        const int32_t IH = is_2D ? src1->ne[1] : 1;
                        const int32_t IW =         src1->ne[0];

                        const int32_t KH = is_2D ? src0->ne[1] : 1;
                        const int32_t KW =         src0->ne[0];

                        const int32_t OH = is_2D ? dst->ne[2] : 1;
                        const int32_t OW =         dst->ne[1];

                        const int32_t CHW = IC * KH * KW;

                        const int32_t ofs0 = src1->nb[is_2D ? 3 : 2] / 4;
                        const int32_t ofs1 = src1->nb[is_2D ? 2 : 1] / 4;

                        id<MTLComputePipelineState> pipeline = nil;

                        switch (dst->type) {
                            case GGML_TYPE_F32: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_IM2COL_F32].pipeline; break;
                            case GGML_TYPE_F16: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_IM2COL_F16].pipeline; break;
                            default: GGML_ASSERT(false);
                        };

                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:id_src1 offset:offs_src1        atIndex:0];
                        [encoder setBuffer:id_dst  offset:offs_dst         atIndex:1];
                        [encoder setBytes:&ofs0    length:sizeof( int32_t) atIndex:2];
                        [encoder setBytes:&ofs1    length:sizeof( int32_t) atIndex:3];
                        [encoder setBytes:&IW      length:sizeof( int32_t) atIndex:4];
                        [encoder setBytes:&IH      length:sizeof( int32_t) atIndex:5];
                        [encoder setBytes:&CHW     length:sizeof( int32_t) atIndex:6];
                        [encoder setBytes:&s0      length:sizeof( int32_t) atIndex:7];
                        [encoder setBytes:&s1      length:sizeof( int32_t) atIndex:8];
                        [encoder setBytes:&p0      length:sizeof( int32_t) atIndex:9];
                        [encoder setBytes:&p1      length:sizeof( int32_t) atIndex:10];
                        [encoder setBytes:&d0      length:sizeof( int32_t) atIndex:11];
                        [encoder setBytes:&d1      length:sizeof( int32_t) atIndex:12];

                        [encoder dispatchThreadgroups:MTLSizeMake(IC, OH, OW) threadsPerThreadgroup:MTLSizeMake(N, KH, KW)];
                    } break;
                case GGML_OP_UPSCALE:
                    {
                        GGML_ASSERT(src0->type == GGML_TYPE_F32);

                        const int sf = dst->op_params[0];

                        const id<MTLComputePipelineState> pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_UPSCALE_F32].pipeline;

                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                        [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                        [encoder setBytes:&ne00 length:sizeof(ne00) atIndex:2];
                        [encoder setBytes:&ne01 length:sizeof(ne01) atIndex:3];
                        [encoder setBytes:&ne02 length:sizeof(ne02) atIndex:4];
                        [encoder setBytes:&ne03 length:sizeof(ne03) atIndex:5];
                        [encoder setBytes:&nb00 length:sizeof(nb00) atIndex:6];
                        [encoder setBytes:&nb01 length:sizeof(nb01) atIndex:7];
                        [encoder setBytes:&nb02 length:sizeof(nb02) atIndex:8];
                        [encoder setBytes:&nb03 length:sizeof(nb03) atIndex:9];
                        [encoder setBytes:&ne0  length:sizeof(ne0)  atIndex:10];
                        [encoder setBytes:&ne1  length:sizeof(ne1)  atIndex:11];
                        [encoder setBytes:&ne2  length:sizeof(ne2)  atIndex:12];
                        [encoder setBytes:&ne3  length:sizeof(ne3)  atIndex:13];
                        [encoder setBytes:&nb0  length:sizeof(nb0)  atIndex:14];
                        [encoder setBytes:&nb1  length:sizeof(nb1)  atIndex:15];
                        [encoder setBytes:&nb2  length:sizeof(nb2)  atIndex:16];
                        [encoder setBytes:&nb3  length:sizeof(nb3)  atIndex:17];
                        [encoder setBytes:&sf   length:sizeof(sf)   atIndex:18];

                        const int nth = MIN((int) pipeline.maxTotalThreadsPerThreadgroup, ne0);

                        [encoder dispatchThreadgroups:MTLSizeMake(ne1, ne2, ne3) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                    } break;
                case GGML_OP_PAD:
                    {
                        GGML_ASSERT(src0->type == GGML_TYPE_F32);

                        id<MTLComputePipelineState> pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_PAD_F32].pipeline;

                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                        [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                        [encoder setBytes:&ne00 length:sizeof(ne00) atIndex:2];
                        [encoder setBytes:&ne01 length:sizeof(ne01) atIndex:3];
                        [encoder setBytes:&ne02 length:sizeof(ne02) atIndex:4];
                        [encoder setBytes:&ne03 length:sizeof(ne03) atIndex:5];
                        [encoder setBytes:&nb00 length:sizeof(nb00) atIndex:6];
                        [encoder setBytes:&nb01 length:sizeof(nb01) atIndex:7];
                        [encoder setBytes:&nb02 length:sizeof(nb02) atIndex:8];
                        [encoder setBytes:&nb03 length:sizeof(nb03) atIndex:9];
                        [encoder setBytes:&ne0  length:sizeof(ne0)  atIndex:10];
                        [encoder setBytes:&ne1  length:sizeof(ne1)  atIndex:11];
                        [encoder setBytes:&ne2  length:sizeof(ne2)  atIndex:12];
                        [encoder setBytes:&ne3  length:sizeof(ne3)  atIndex:13];
                        [encoder setBytes:&nb0  length:sizeof(nb0)  atIndex:14];
                        [encoder setBytes:&nb1  length:sizeof(nb1)  atIndex:15];
                        [encoder setBytes:&nb2  length:sizeof(nb2)  atIndex:16];
                        [encoder setBytes:&nb3  length:sizeof(nb3)  atIndex:17];

                        const int nth = MIN(1024, ne0);

                        [encoder dispatchThreadgroups:MTLSizeMake(ne1, ne2, ne3) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                    } break;
                case GGML_OP_ARANGE:
                    {
                        GGML_ASSERT(dst->type == GGML_TYPE_F32);

                        float start;
                        float step;

                        memcpy(&start, ((int32_t *) dst->op_params) + 0, sizeof(float));
                        memcpy(&step,  ((int32_t *) dst->op_params) + 2, sizeof(float));

                        id<MTLComputePipelineState> pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_ARANGE_F32].pipeline;

                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:id_dst  offset:offs_dst    atIndex:0];
                        [encoder setBytes:&ne0   length:sizeof(ne0)   atIndex:1];
                        [encoder setBytes:&start length:sizeof(start) atIndex:2];
                        [encoder setBytes:&step  length:sizeof(step)  atIndex:3];

                        const int nth = MIN(1024, ne0);

                        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                    } break;
                case GGML_OP_TIMESTEP_EMBEDDING:
                    {
                        GGML_ASSERT(src0->type == GGML_TYPE_F32);

                        const int dim        = dst->op_params[0];
                        const int max_period = dst->op_params[1];

                        const int half = dim / 2;

                        id<MTLComputePipelineState> pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_TIMESTEP_EMBEDDING_F32].pipeline;

                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                        [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                        [encoder setBytes:&nb1   length:sizeof(nb1) atIndex:2];
                        [encoder setBytes:&dim   length:sizeof(dim) atIndex:3];
                        [encoder setBytes:&max_period length:sizeof(max_period) atIndex:4];

                        const int nth = MIN(1024, half);

                        [encoder dispatchThreadgroups:MTLSizeMake(ne00, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                    } break;
                case GGML_OP_ARGSORT:
                    {
                        GGML_ASSERT(src0->type == GGML_TYPE_F32);
                        GGML_ASSERT( dst->type == GGML_TYPE_I32);

                        const int nrows = ggml_nrows(src0);

                        enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

                        // bitonic sort requires the number of elements to be power of 2
                        int64_t ne00_padded = 1;
                        while (ne00_padded < ne00) {
                            ne00_padded *= 2;
                        }

                        // Metal kernels require the buffer size to be multiple of 16 bytes
                        // https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/1443142-setthreadgroupmemorylength
                        const int mem_size = GGML_PAD(ne00_padded*sizeof(int32_t), 16);

                        id<MTLComputePipelineState> pipeline = nil;

                        switch (order) {
                            case GGML_SORT_ORDER_ASC:  pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_ARGSORT_F32_I32_ASC].pipeline;  break;
                            case GGML_SORT_ORDER_DESC: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_ARGSORT_F32_I32_DESC].pipeline; break;
                            default: GGML_ASSERT(false);
                        };

                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:id_src0     offset:offs_src0        atIndex:0];
                        [encoder setBuffer:id_dst      offset:offs_dst         atIndex:1];
                        [encoder setBytes:&ne00        length:sizeof( int64_t) atIndex:2];
                        [encoder setBytes:&ne00_padded length:sizeof( int64_t) atIndex:3];
                        [encoder setThreadgroupMemoryLength:mem_size atIndex:0];

                        [encoder dispatchThreadgroups:MTLSizeMake(1, nrows, 1) threadsPerThreadgroup:MTLSizeMake(ne00_padded, 1, 1)];
                    } break;
                case GGML_OP_LEAKY_RELU:
                    {
                        GGML_ASSERT(src0->type == GGML_TYPE_F32);

                        float slope;
                        memcpy(&slope, dst->op_params, sizeof(float));

                        id<MTLComputePipelineState> pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_LEAKY_RELU_F32].pipeline;

                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:id_src0 offset:offs_src0   atIndex:0];
                        [encoder setBuffer:id_dst  offset:offs_dst    atIndex:1];
                        [encoder setBytes:&slope length:sizeof(slope) atIndex:2];

                        const int64_t n = ggml_nelements(dst);

                        [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                    } break;
                case GGML_OP_DUP:
                case GGML_OP_CPY:
                case GGML_OP_CONT:
                    {
                        GGML_ASSERT(ne00 % ggml_blck_size(src0->type) == 0);

                        int nth = MIN(1024, ne00/ggml_blck_size(src0->type));

                        id<MTLComputePipelineState> pipeline = nil;

                        switch (src0t) {
                            case GGML_TYPE_F32:
                                {
                                    GGML_ASSERT(ne0 % ggml_blck_size(dst->type) == 0);

                                    switch (dstt) {
                                        case GGML_TYPE_F16:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_CPY_F32_F16].pipeline;  break;
                                        case GGML_TYPE_F32:    pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_CPY_F32_F32].pipeline;  break;
                                        case GGML_TYPE_Q8_0:   pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_CPY_F32_Q8_0].pipeline; break;
                                        case GGML_TYPE_Q4_0:   pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_CPY_F32_Q4_0].pipeline; break;
                                        case GGML_TYPE_Q4_1:   pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_CPY_F32_Q4_1].pipeline; break;
                                        case GGML_TYPE_Q5_0:   pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_CPY_F32_Q5_0].pipeline; break;
                                        case GGML_TYPE_Q5_1:   pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_CPY_F32_Q5_1].pipeline; break;
                                        case GGML_TYPE_IQ4_NL: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_CPY_F32_IQ4_NL].pipeline; break;
                                        default: GGML_ASSERT(false && "not implemented");
                                    };
                                } break;
                            case GGML_TYPE_F16:
                                {
                                    switch (dstt) {
                                        case GGML_TYPE_F16: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_CPY_F16_F16].pipeline; break;
                                        case GGML_TYPE_F32: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_CPY_F16_F32].pipeline; break;
                                        default: GGML_ASSERT(false && "not implemented");
                                    };
                                } break;
                            default: GGML_ASSERT(false && "not implemented");
                        }

                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:id_src0 offset:offs_src0        atIndex:0];
                        [encoder setBuffer:id_dst  offset:offs_dst         atIndex:1];
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

                        [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                    } break;
                default:
                    {
                        GGML_METAL_LOG_ERROR("%s: error: node %3d, op = %8s not implemented\n", __func__, i, ggml_op_name(dst->op));
                        GGML_ASSERT(false);
                    }
            }

            if (should_capture) {
                [encoder popDebugGroup];
            }
        }

        [encoder endEncoding];

        [command_buffer commit];
    });

    // Wait for completion and check status of each command buffer
    // needed to detect if the device ran out-of-memory for example (#1881)

    for (int i = 0; i < n_cb; ++i) {
        id<MTLCommandBuffer> command_buffer = command_buffers[i];
        [command_buffer waitUntilCompleted];

        MTLCommandBufferStatus status = [command_buffer status];
        if (status != MTLCommandBufferStatusCompleted) {
            GGML_METAL_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, i, status);
            return GGML_STATUS_FAILED;
        }
    }

    if (should_capture) {
        [[MTLCaptureManager sharedCaptureManager] stopCapture];
    }

    }
    return GGML_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////

// backend interface

// default buffer
static id<MTLDevice> g_backend_device = nil;
static int g_backend_device_ref_count = 0;

static id<MTLDevice> ggml_backend_metal_get_device(void) {
    if (g_backend_device == nil) {
        g_backend_device = MTLCreateSystemDefaultDevice();
    }

    g_backend_device_ref_count++;

    return g_backend_device;
}

static void ggml_backend_metal_free_device(void) {
    assert(g_backend_device_ref_count > 0);

    g_backend_device_ref_count--;

    if (g_backend_device_ref_count == 0) {
        [g_backend_device release];
        g_backend_device = nil;
    }
}

GGML_CALL static const char * ggml_backend_metal_buffer_get_name(ggml_backend_buffer_t buffer) {
    return "Metal";

    UNUSED(buffer);
}

GGML_CALL static void ggml_backend_metal_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    struct ggml_backend_metal_buffer_context * ctx = (struct ggml_backend_metal_buffer_context *)buffer->context;

    for (int i = 0; i < ctx->n_buffers; i++) {
        [ctx->buffers[i].metal release];
    }
    ggml_backend_metal_free_device();

    if (ctx->owned) {
        free(ctx->all_data);
    }

    free(ctx);
}

GGML_CALL static void * ggml_backend_metal_buffer_get_base(ggml_backend_buffer_t buffer) {
    struct ggml_backend_metal_buffer_context * ctx = (struct ggml_backend_metal_buffer_context *)buffer->context;

    return ctx->all_data;
}

GGML_CALL static void ggml_backend_metal_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *)tensor->data + offset, data, size);

    UNUSED(buffer);
}

GGML_CALL static void ggml_backend_metal_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *)tensor->data + offset, size);

    UNUSED(buffer);
}

GGML_CALL static bool ggml_backend_metal_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    UNUSED(buffer);
}

GGML_CALL static void ggml_backend_metal_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    struct ggml_backend_metal_buffer_context * ctx = (struct ggml_backend_metal_buffer_context *)buffer->context;

    memset(ctx->all_data, value, ctx->all_size);
}

static struct ggml_backend_buffer_i ggml_backend_metal_buffer_i = {
    /* .get_name        = */ ggml_backend_metal_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_metal_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_metal_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .set_tensor      = */ ggml_backend_metal_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_metal_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_metal_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_metal_buffer_clear,
    /* .reset           = */ NULL,
};

// default buffer type

GGML_CALL static const char * ggml_backend_metal_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "Metal";

    UNUSED(buft);
}

static void ggml_backend_metal_log_allocated_size(id<MTLDevice> device) {
#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
    if (@available(macOS 10.12, iOS 16.0, *)) {
        GGML_METAL_LOG_INFO(", (%8.2f / %8.2f)",
                device.currentAllocatedSize / 1024.0 / 1024.0,
                device.recommendedMaxWorkingSetSize / 1024.0 / 1024.0);

        if (device.currentAllocatedSize > device.recommendedMaxWorkingSetSize) {
            GGML_METAL_LOG_WARN("%s: warning: current allocated size is greater than the recommended max working set size\n", __func__);
        } else {
            GGML_METAL_LOG_INFO("\n");
        }
    } else {
        GGML_METAL_LOG_INFO(", (%8.2f)\n", device.currentAllocatedSize / 1024.0 / 1024.0);
    }
#endif
    UNUSED(device);
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_metal_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    struct ggml_backend_metal_buffer_context * ctx = malloc(sizeof(struct ggml_backend_metal_buffer_context));

    const size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    id<MTLDevice> device = ggml_backend_metal_get_device();

    ctx->all_data = ggml_metal_host_malloc(size_aligned);
    ctx->all_size = size_aligned;
    ctx->owned = true;
    ctx->n_buffers = 1;

    ctx->buffers[0].data = ctx->all_data;
    ctx->buffers[0].size = size;
    ctx->buffers[0].metal = [device newBufferWithBytesNoCopy:ctx->all_data
                    length:size_aligned
                    options:MTLResourceStorageModeShared
                    deallocator:nil];

    if (ctx->buffers[0].metal == nil) {
        GGML_METAL_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
        free(ctx);
        ggml_backend_metal_free_device();
        return NULL;
    }

    GGML_METAL_LOG_INFO("%s: allocated buffer, size = %8.2f MiB", __func__, size_aligned / 1024.0 / 1024.0);
    ggml_backend_metal_log_allocated_size(device);

    return ggml_backend_buffer_init(buft, ggml_backend_metal_buffer_i, ctx, size);
}

GGML_CALL static size_t ggml_backend_metal_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;
    UNUSED(buft);
}

GGML_CALL static size_t ggml_backend_metal_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    id<MTLDevice> device = ggml_backend_metal_get_device();
    size_t max_size = device.maxBufferLength;
    ggml_backend_metal_free_device();

    return max_size;

    UNUSED(buft);
}

GGML_CALL static bool ggml_backend_metal_buffer_type_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    return ggml_backend_is_metal(backend) || ggml_backend_is_cpu(backend);

    UNUSED(buft);
}

GGML_CALL static bool ggml_backend_metal_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    UNUSED(buft);
}

GGML_CALL ggml_backend_buffer_type_t ggml_backend_metal_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_buffer_type_metal = {
        /* .iface = */ {
            /* .get_name         = */ ggml_backend_metal_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_metal_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_metal_buffer_type_get_alignment,
            /* .get_max_size     = */ ggml_backend_metal_buffer_type_get_max_size,
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .supports_backend = */ ggml_backend_metal_buffer_type_supports_backend,
            /* .is_host          = */ ggml_backend_metal_buffer_type_is_host,
        },
        /* .context = */ NULL,
    };

    return &ggml_backend_buffer_type_metal;
}

// buffer from ptr

GGML_CALL ggml_backend_buffer_t ggml_backend_metal_buffer_from_ptr(void * data, size_t size, size_t max_size) {
    struct ggml_backend_metal_buffer_context * ctx = malloc(sizeof(struct ggml_backend_metal_buffer_context));

    ctx->all_data = data;
    ctx->all_size = size;
    ctx->owned = false;
    ctx->n_buffers = 0;

    const size_t size_page = sysconf(_SC_PAGESIZE);

    // page-align the data ptr
    {
        const uintptr_t offs = (uintptr_t) data % size_page;
        data  = (void *) ((char *) data - offs);
        size += offs;
    }

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    id<MTLDevice> device = ggml_backend_metal_get_device();

    // the buffer fits into the max buffer size allowed by the device
    if (size_aligned <= device.maxBufferLength) {
        ctx->buffers[ctx->n_buffers].data = data;
        ctx->buffers[ctx->n_buffers].size = size;

        ctx->buffers[ctx->n_buffers].metal = [device newBufferWithBytesNoCopy:data length:size_aligned options:MTLResourceStorageModeShared deallocator:nil];

        if (ctx->buffers[ctx->n_buffers].metal == nil) {
            GGML_METAL_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
            return false;
        }

        GGML_METAL_LOG_INFO("%s: allocated buffer, size = %8.2f MiB", __func__, size_aligned / 1024.0 / 1024.0);

        ++ctx->n_buffers;
    } else {
        // this overlap between the views will guarantee that the tensor with the maximum size will fully fit into
        // one of the views
        const size_t size_ovlp = ((max_size + size_page - 1) / size_page + 1) * size_page; // round-up 2 pages just in case
        const size_t size_step = device.maxBufferLength - size_ovlp;
        const size_t size_view = device.maxBufferLength;

        for (size_t i = 0; i < size; i += size_step) {
            const size_t size_step_aligned = (i + size_view <= size) ? size_view : (size_aligned - i);

            ctx->buffers[ctx->n_buffers].data = (void *) ((uint8_t *) data + i);
            ctx->buffers[ctx->n_buffers].size = size_step_aligned;

            ctx->buffers[ctx->n_buffers].metal = [device newBufferWithBytesNoCopy:(void *) ((uint8_t *) data + i) length:size_step_aligned options:MTLResourceStorageModeShared deallocator:nil];

            if (ctx->buffers[ctx->n_buffers].metal == nil) {
                GGML_METAL_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_step_aligned / 1024.0 / 1024.0);
                return false;
            }

            GGML_METAL_LOG_INFO("%s: allocated buffer, size = %8.2f MiB, offs = %12ld", __func__, size_step_aligned / 1024.0 / 1024.0, i);
            if (i + size_step < size) {
                GGML_METAL_LOG_INFO("\n");
            }

            ++ctx->n_buffers;
        }
    }

    ggml_backend_metal_log_allocated_size(device);

    return ggml_backend_buffer_init(ggml_backend_metal_buffer_type(), ggml_backend_metal_buffer_i, ctx, size);
}

// backend

GGML_CALL static const char * ggml_backend_metal_name(ggml_backend_t backend) {
    return "Metal";

    UNUSED(backend);
}

GGML_CALL static void ggml_backend_metal_free(ggml_backend_t backend) {
    struct ggml_metal_context * ctx = (struct ggml_metal_context *)backend->context;
    ggml_metal_free(ctx);
    free(backend);
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_metal_get_default_buffer_type(ggml_backend_t backend) {
    return ggml_backend_metal_buffer_type();

    UNUSED(backend);
}

GGML_CALL static enum ggml_status ggml_backend_metal_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_metal_context * metal_ctx = (struct ggml_metal_context *)backend->context;

    return ggml_metal_graph_compute(metal_ctx, cgraph);
}

GGML_CALL static bool ggml_backend_metal_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    struct ggml_metal_context * metal_ctx = (struct ggml_metal_context *)backend->context;

    return ggml_metal_supports_op(metal_ctx, op);
}

static struct ggml_backend_i ggml_backend_metal_i = {
    /* .get_name                = */ ggml_backend_metal_name,
    /* .free                    = */ ggml_backend_metal_free,
    /* .get_default_buffer_type = */ ggml_backend_metal_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_metal_graph_compute,
    /* .supports_op             = */ ggml_backend_metal_supports_op,
    /* .offload_op              = */ NULL,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

void ggml_backend_metal_log_set_callback(ggml_log_callback log_callback, void * user_data) {
    ggml_metal_log_callback  = log_callback;
    ggml_metal_log_user_data = user_data;
}

static ggml_guid_t ggml_backend_metal_guid(void) {
    static ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    return &guid;
}

ggml_backend_t ggml_backend_metal_init(void) {
    struct ggml_metal_context * ctx = ggml_metal_init(GGML_DEFAULT_N_THREADS);

    if (ctx == NULL) {
        return NULL;
    }

    ggml_backend_t metal_backend = malloc(sizeof(struct ggml_backend));

    *metal_backend = (struct ggml_backend) {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .context   = */ ctx,
    };

    return metal_backend;
}

bool ggml_backend_is_metal(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_metal_guid());
}

void ggml_backend_metal_set_n_cb(ggml_backend_t backend, int n_cb) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    struct ggml_metal_context * ctx = (struct ggml_metal_context *)backend->context;

    ctx->n_cb = MIN(n_cb, GGML_METAL_MAX_BUFFERS);
}

bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    struct ggml_metal_context * ctx = (struct ggml_metal_context *)backend->context;

    return [ctx->device supportsFamily:(MTLGPUFamilyApple1 + family - 1)];
}

void ggml_backend_metal_capture_next_compute(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    struct ggml_metal_context * ctx = (struct ggml_metal_context *)backend->context;
    ctx->should_capture_next_compute = true;
}

GGML_CALL ggml_backend_t ggml_backend_reg_metal_init(const char * params, void * user_data); // silence warning

GGML_CALL ggml_backend_t ggml_backend_reg_metal_init(const char * params, void * user_data) {
    return ggml_backend_metal_init();

    GGML_UNUSED(params);
    GGML_UNUSED(user_data);
}
