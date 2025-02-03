#include "clip.h"
#include "llava.h"

#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#if defined(LLAVA_LOG_OFF)
#   define LOG_INF(...)
#   define LOG_WRN(...)
#   define LOG_ERR(...)
#   define LOG_DBG(...)
#else // defined(LLAVA_LOG_OFF)
#   define LOG_INF(...) do { fprintf(stdout, __VA_ARGS__); } while (0)
#   define LOG_WRN(...) do { fprintf(stderr, __VA_ARGS__); } while (0)
#   define LOG_ERR(...) do { fprintf(stderr, __VA_ARGS__); } while (0)
#   define LOG_DBG(...) do { fprintf(stdout, __VA_ARGS__); } while (0)
#endif // defined(LLAVA_LOG_OFF)

// RGB uint8 image
struct clip_image_u8 {
    int nx;
    int ny;

    std::vector<uint8_t> buf;
};

// RGB float32 image (NHWC)
// Memory layout: RGBRGBRGB...
struct clip_image_f32 {
    int nx;
    int ny;

    std::vector<float> buf;
};

struct clip_image_grid_shape {
    int first;
    int second;
};

/**
 * Selects the best resolution from a list of possible resolutions based on the original size.
 *
 * @param original_size The original size of the image in the format (width, height).
 * @param possible_resolutions A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].
 * @return The best fit resolution in the format (width, height).
 */
static std::pair<int, int> select_best_resolution(const std::pair<int, int>& original_size, const std::vector<std::pair<int, int>>& possible_resolutions) {
    int original_width  = original_size.first;
    int original_height = original_size.second;

    std::pair<int, int> best_fit;
    int max_effective_resolution = 0;
    int min_wasted_resolution = std::numeric_limits<int>::max();

    for (const auto& resolution : possible_resolutions) {
        int width = resolution.first;
        int height = resolution.second;
        float scale = std::min(static_cast<float>(width) / original_width, static_cast<float>(height) / original_height);
        int downscaled_width  = static_cast<int>(original_width * scale);
        int downscaled_height = static_cast<int>(original_height * scale);
        int effective_resolution = std::min(downscaled_width * downscaled_height, original_width * original_height);
        int wasted_resolution = (width * height) - effective_resolution;
        // LOG_DBG("resolution: %d %d, scale: %f, downscaled: %d %d, effective: %d, wasted: %d\n", width, height, scale, downscaled_width, downscaled_height, effective_resolution, wasted_resolution);
        if (effective_resolution > max_effective_resolution || (effective_resolution == max_effective_resolution && wasted_resolution < min_wasted_resolution)) {
            max_effective_resolution = effective_resolution;
            min_wasted_resolution = wasted_resolution;
            best_fit = resolution;
        }
    }

    return best_fit;
}

/**
 * @brief Get the anyres image grid shape object
 *
 * @param image_size
 * @param grid_pinpoints
 * @param image_patch_size
 * @return <int, int>
 */
static struct clip_image_grid_shape get_anyres_image_grid_shape(const std::pair<int, int> & image_size, const std::vector<std::pair<int, int>> & grid_pinpoints, int image_patch_size) {
    /**
        Conversion from gguf flat array to vector:
        std::vector<std::pair<int, int>> possible_resolutions;
        for (int i = 0; i < 32 && params.image_grid_pinpoints[i] != 0; i+=2) {
            possible_resolutions.push_back({params.image_grid_pinpoints[i], params.image_grid_pinpoints[i+1]});
        }
     */
    auto best_resolution = select_best_resolution(image_size, grid_pinpoints);
    return {best_resolution.first / image_patch_size, best_resolution.second / image_patch_size};
}

// Take the image segments in a grid configuration and return the embeddings and the number of embeddings into preallocated memory (image_embd_out)
static bool clip_llava_handle_patches(clip_ctx * ctx_clip, std::vector<float *> & image_embd_v, struct clip_image_grid_shape grid_shape, float * image_embd_out, int * n_img_pos_out) {
    struct {
        struct ggml_context * ctx;
    } model;

    const int32_t image_size = clip_image_size(ctx_clip);
    const int32_t patch_size = clip_patch_size(ctx_clip);

    int32_t num_patches_per_side = image_size / patch_size; // 336 / 14 = 24 - used for embedding-patching boxes (24*24 = 576 patches)

    int num_patches_width  = grid_shape.first;  // grid 1-4
    int num_patches_height = grid_shape.second; // grid 1-4

    const size_t num_images = num_patches_width * num_patches_height + 1;

    // TODO: size calculation is not calculated - it's only tens of MB
    size_t ctx_size = 0;

    {
        ctx_size += clip_embd_nbytes(ctx_clip) * num_images * 8; // image_features
        ctx_size += 1024*1024 * ggml_type_size(GGML_TYPE_F32);
    }

    struct ggml_init_params params {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false, // NOTE: this should be false when using the legacy API
    };

    // Python reference code for full unpad:
    /*
        base_image_feature = image_feature[0]
        image_feature = image_feature[1:]
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = unpad_image(image_feature, image_sizes[image_idx])
        image_feature = torch.cat((
            image_feature,
            self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1)
        ), dim=-1)
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
    */
    // We now have two options: unpad or no unpad. Unpad removes tokens for faster llm eval.
    // In terms of result quality it appears to make no difference, so we'll start with the easier approach given 5D tensors are not supported in ggml yet.
    // Without unpad we have to split the sub-image embeddings into patches of 24 features each and permute them.
    // Once all images are processed to prepended the base_image_features without any changes.

    // Pytorch reference simplified, modified for ggml compatibility - confirmed identical output in python (for a 2x2 grid image (676x676 scaling))
    /*
        image_feature = image_feature.view(2, 2, 24, 24, 4096)
        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
        image_feature = image_feature.view(2, 24, 2, 24, 4096)
        image_feature = image_feature.flatten(0, 3)

        // Reshape to 4D tensor by merging the last two dimensions
        image_feature = image_feature.view(2, 2, 24, 24*4096)
        image_feature = image_feature.permute(0, 2, 1, 3).contiguous()
        image_feature = image_feature.view(-1, 4096)
    */

    model.ctx = ggml_init(params);

    struct ggml_tensor * image_features = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F32, clip_n_mmproj_embd(ctx_clip), clip_n_patches(ctx_clip), num_images - 1); // example: 4096 x 576 x 4
    // ggml_tensor_printf(image_features,"image_features",__LINE__,false,false);
    // fill it with the image embeddings, ignoring the base
    for (size_t i = 1; i < num_images; i++) {
        size_t offset = (i-1) * clip_embd_nbytes(ctx_clip);
        memcpy((uint8_t *)(image_features->data) + offset, image_embd_v[i], clip_embd_nbytes(ctx_clip));
    }

    struct ggml_cgraph  * gf = ggml_new_graph(model.ctx);
    size_t size_ele = ggml_type_size(GGML_TYPE_F32);

    struct ggml_tensor *image_features_patchview = ggml_view_4d(model.ctx, image_features,
                                                                num_patches_per_side * clip_n_mmproj_embd(ctx_clip),
                                                                num_patches_per_side,
                                                                num_patches_width,
                                                                num_patches_height,
                                                                size_ele * num_patches_per_side * clip_n_mmproj_embd(ctx_clip),
                                                                size_ele * num_patches_per_side * clip_n_mmproj_embd(ctx_clip) * num_patches_per_side,
                                                                size_ele * num_patches_per_side * clip_n_mmproj_embd(ctx_clip) * num_patches_per_side * num_patches_width, 0);
    // ggml_tensor_printf(image_features_patchview,"image_features_patchview",__LINE__,false,false);
    struct ggml_tensor *permuted_cont = ggml_cont(model.ctx, ggml_permute(model.ctx, image_features_patchview, 0, 2, 1, 3));
    /**
     At the end of each row we have to add the row_end embeddings, which are the same as the newline embeddings
         image_feature = torch.cat((
        image_feature,
        self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
    ), dim=-1)
     *
     */

    // ggml_tensor_printf(permuted_cont,"permuted_cont",__LINE__,false,false);
    struct ggml_tensor *flatten = ggml_view_2d(model.ctx, permuted_cont, clip_n_mmproj_embd(ctx_clip), num_patches_height * num_patches_width * num_patches_per_side * num_patches_per_side,  size_ele * clip_n_mmproj_embd(ctx_clip), 0);
    // ggml_tensor_printf(flatten,"flatten",__LINE__,false,false);
    ggml_build_forward_expand(gf, flatten);
    ggml_graph_compute_with_ctx(model.ctx, gf, 1);
    struct ggml_tensor* result = ggml_graph_node(gf, -1);

    memcpy(image_embd_out, image_embd_v[0], clip_embd_nbytes(ctx_clip)); // main image as global context
    // append without newline tokens (default behavior in llava_arch when not using unpad ):
    memcpy(image_embd_out + clip_n_patches(ctx_clip) * clip_n_mmproj_embd(ctx_clip), (float*)result->data, clip_embd_nbytes(ctx_clip) * (num_images-1)); // grid patches
    *n_img_pos_out = static_cast<int>(result->ne[1]+clip_n_patches(ctx_clip));

    // Debug: Test single segments
    // Current findings: sending base image, sending a segment embedding all works similar to python
    // However, permuted embeddings do not work yet (stride issue?)
    // memcpy(image_embd_out, image_embd_v[0], clip_embd_nbytes(ctx_clip)); // main image as context
    // memcpy(image_embd_out, (float*)prepared_cont->data, clip_embd_nbytes(ctx_clip)); // main image as context
    // *n_img_pos_out=576;

    ggml_free(model.ctx);
    return true;
}

static clip_image_f32 * only_v2_5_reshape_by_patch(clip_image_f32 * image, int patch_size) {
    int width = image->nx;
    int height = image->ny;
    int num_patches = (height / patch_size) * (width / patch_size);
    clip_image_f32 * patch = clip_image_f32_init();
    patch->nx = patch_size * num_patches;
    patch->ny = patch_size;
    patch->buf.resize(3 * patch->nx * patch->ny);

    int patch_index = 0;

    for (int i = 0; i < height; i += patch_size) {
        for (int j = 0; j < width; j += patch_size) {
            for (int pi = 0; pi < patch_size; ++pi) {
                for (int pj = 0; pj < patch_size; ++pj) {
                    int input_index = ((i + pi) * width + (j + pj)) * 3;
                    int output_index = (pi * patch_size * num_patches + patch_index * patch_size + pj) * 3;
                    patch->buf[output_index] = image->buf[input_index];
                    patch->buf[output_index+1] = image->buf[input_index+1];
                    patch->buf[output_index+2] = image->buf[input_index+2];
                }
            }
            patch_index++;
        }
    }
    return patch;
}

static bool encode_image_with_clip(clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float * image_embd, int * n_img_pos) {
    // std::vector<clip_image_f32*> img_res_v; // format VectN x H x W x RGB (N x 336 x 336 x 3), so interleaved RGB - different to the python implementation which is N x 3 x 336 x 336
    clip_image_f32_batch img_res_v;
    img_res_v.size = 0;
    img_res_v.data = nullptr;
    if (!clip_image_preprocess(ctx_clip, img, &img_res_v)) {
        LOG_ERR("%s: unable to preprocess image\n", __func__);
        delete[] img_res_v.data;
        return false;
    }

    const int64_t t_img_enc_start_us = ggml_time_us();

    const char * mm_patch_merge_type = clip_patch_merge_type(ctx_clip);

    if (clip_is_minicpmv(ctx_clip) || clip_is_qwen2vl(ctx_clip)) {
        std::vector<float *> image_embd_v;
        image_embd_v.resize(img_res_v.size);
        struct clip_image_size * load_image_size = clip_image_size_init();

        for (size_t i = 0; i < img_res_v.size; i++) {
            const int64_t t_img_enc_step_start_us = ggml_time_us();
            image_embd_v[i] = (float *)malloc(clip_embd_nbytes_by_img(ctx_clip, img_res_v.data[i].nx, img_res_v.data[i].ny));
            int patch_size=14;
            load_image_size->width = img_res_v.data[i].nx;
            load_image_size->height = img_res_v.data[i].ny;
            clip_add_load_image_size(ctx_clip, load_image_size);

            bool encoded = false;
            if (clip_is_qwen2vl(ctx_clip)) {
                encoded = clip_image_encode(ctx_clip, n_threads, &img_res_v.data[i], image_embd_v[i]);
            }
            else {
                int has_minicpmv_projector = clip_is_minicpmv(ctx_clip);
                if (has_minicpmv_projector == 2) {
                    encoded = clip_image_encode(ctx_clip, n_threads, only_v2_5_reshape_by_patch(&img_res_v.data[i], patch_size), image_embd_v[i]);
                }
                else if (has_minicpmv_projector == 3) {
                    encoded = clip_image_encode(ctx_clip, n_threads, &img_res_v.data[i], image_embd_v[i]);
                }
            }

            if (!encoded) {
                LOG_ERR("Unable to encode image - spatial_unpad - subimage %d of %d\n", (int) i+1, (int) img_res_v.size);
                return false;
            }
            const int64_t t_img_enc_steop_batch_us = ggml_time_us();
            LOG_INF("%s: step %d of %d encoded in %8.2f ms\n", __func__, (int)i+1, (int)img_res_v.size, (t_img_enc_steop_batch_us - t_img_enc_step_start_us) / 1000.0);
        }
        const int64_t t_img_enc_batch_us = ggml_time_us();
        LOG_INF("%s: all %d segments encoded in %8.2f ms\n", __func__, (int)img_res_v.size, (t_img_enc_batch_us - t_img_enc_start_us) / 1000.0);

        int n_img_pos_out = 0;
        for (size_t i = 0; i < image_embd_v.size(); i++) {
            std::memcpy(
                image_embd + n_img_pos_out * clip_n_mmproj_embd(ctx_clip),
                image_embd_v[i],
                clip_embd_nbytes_by_img(ctx_clip, img_res_v.data[i].nx, img_res_v.data[i].ny));
            n_img_pos_out += clip_n_patches_by_img(ctx_clip, &img_res_v.data[i]);
        }
        *n_img_pos = n_img_pos_out;
        for (size_t i = 0; i < image_embd_v.size(); i++) {
            free(image_embd_v[i]);
        }
        image_embd_v.clear();
        load_image_size->width = img->nx;
        load_image_size->height = img->ny;
        clip_add_load_image_size(ctx_clip, load_image_size);
        LOG_INF("%s: load_image_size %d %d\n", __func__, load_image_size->width, load_image_size->height);
    }
    else if (strcmp(mm_patch_merge_type, "spatial_unpad") != 0) {
        // flat / default llava-1.5 type embedding
        *n_img_pos = clip_n_patches(ctx_clip);
        bool encoded = clip_image_encode(ctx_clip, n_threads, &img_res_v.data[0], image_embd); // image_embd shape is 576 x 4096
        delete[] img_res_v.data;
        if (!encoded) {
            LOG_ERR("Unable to encode image\n");

            return false;
        }
    }
    else {
        // spatial_unpad llava-1.6 type embedding
        // TODO: CLIP needs batching support - in HF the llm projection is separate after encoding, which might be a solution to quickly get batching working
        std::vector<float *> image_embd_v;
        image_embd_v.resize(img_res_v.size);
        for (size_t i = 0; i < img_res_v.size; i++) {
            image_embd_v[i] = (float *)malloc(clip_embd_nbytes(ctx_clip)); // 576 patches * 4096 embeddings * 4 bytes = 9437184
            const bool encoded = clip_image_encode(ctx_clip, n_threads, &img_res_v.data[i], image_embd_v[i]); // image data is in 3x336x336 format and will be converted to 336x336x3 inside
            if (!encoded) {
                LOG_ERR("Unable to encode image - spatial_unpad - subimage %d of %d\n", (int) i+1, (int) img_res_v.size);
                return false;
            }
        }
        const int64_t t_img_enc_batch_us = ggml_time_us();
        LOG_INF("%s: %d segments encoded in %8.2f ms\n", __func__, (int)img_res_v.size, (t_img_enc_batch_us - t_img_enc_start_us) / 1000.0);

        const int32_t * image_grid = clip_image_grid(ctx_clip);

        std::vector<std::pair<int, int>> grid_pinpoints;
        for (int i = 0; i < 32 && image_grid[i] != 0; i += 2) {
            grid_pinpoints.push_back({image_grid[i], image_grid[i+1]});
        }

        // free all img_res_v - not needed anymore
        delete[] img_res_v.data;
        img_res_v.size = 0;
        img_res_v.data = nullptr;

        const int32_t image_size = clip_image_size(ctx_clip);

        struct clip_image_grid_shape grid_shape = get_anyres_image_grid_shape({img->nx,img->ny}, grid_pinpoints, image_size);

        int n_img_pos_out;
        clip_llava_handle_patches(ctx_clip, image_embd_v, grid_shape, image_embd, &n_img_pos_out);
        *n_img_pos = n_img_pos_out;

        for (size_t i = 0; i < image_embd_v.size(); i++) {
            free(image_embd_v[i]);
        }
        image_embd_v.clear();

        // debug image/segment/normalization content:
        // clip_image_u8 * tmp = clip_image_u8_init();
        // clip_image_convert_f32_to_u8(*image_feature, *tmp);
        // clip_image_save_to_bmp(*tmp, "image_feature.bmp");
    }

    LOG_INF("%s: image embedding created: %d tokens\n", __func__, *n_img_pos);

    const int64_t t_img_enc_end_us = ggml_time_us();
    float t_img_enc_ms = (t_img_enc_end_us - t_img_enc_start_us) / 1000.0;

    LOG_INF("\n%s: image encoded in %8.2f ms by CLIP (%8.2f ms per image patch)\n", __func__, t_img_enc_ms, t_img_enc_ms / *n_img_pos);

    return true;
}

bool llava_validate_embed_size(const llama_context * ctx_llama, const clip_ctx * ctx_clip) {
        // make sure that the correct mmproj was used, i.e., compare apples to apples
    int n_llama_embd = llama_n_embd(llama_get_model(ctx_llama));
    auto n_image_embd = clip_n_mmproj_embd(ctx_clip);
    if (n_image_embd != n_llama_embd) {
        LOG_ERR("%s: embedding dim of the multimodal projector (%d) is not equal to that of LLaMA (%d). Make sure that you use the correct mmproj file.\n", __func__, n_image_embd, n_llama_embd);
        return false;
    }
    return true;
}

bool llava_image_embed_make_with_clip_img(clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float ** image_embd_out, int * n_img_pos_out) {
    int num_max_patches = 6;
    if (clip_is_minicpmv(ctx_clip)) {
        num_max_patches = 10;
    }
    float * image_embd;
    if (clip_is_qwen2vl(ctx_clip)) {
        // qwen2vl don't split image into chunks, so `num_max_patches` is not needed.
        image_embd = (float *)malloc(clip_embd_nbytes_by_img(ctx_clip, img->nx, img->ny));
    } else {
        image_embd = (float *)malloc(clip_embd_nbytes(ctx_clip)*num_max_patches); // TODO: base on gridsize/llava model
    }
    if (!image_embd) {
        LOG_ERR("Unable to allocate memory for image embeddings\n");
        return false;
    }

    int n_img_pos;
    if (!encode_image_with_clip(ctx_clip, n_threads, img, image_embd, &n_img_pos)) {
        LOG_ERR("%s: cannot encode image, aborting\n", __func__);
        free(image_embd);
        return false;
    }
    *image_embd_out = image_embd;
    *n_img_pos_out = n_img_pos;

    return true;
}

struct llava_embd_batch {
    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t>         logits;
    llama_batch batch;
    llava_embd_batch(float * embd, int32_t n_embd, int32_t n_tokens, llama_pos pos_0, llama_seq_id seq_id) {
        pos     .resize(n_tokens);
        n_seq_id.resize(n_tokens);
        seq_ids .resize(n_tokens + 1);
        logits  .resize(n_tokens);
        seq_id_0.resize(1);
        seq_id_0[0] = seq_id;
        seq_ids [n_tokens] = nullptr;
        batch = {
            /*n_tokens       =*/ n_tokens,
            /*tokens         =*/ nullptr,
            /*embd           =*/ embd,
            /*n_embd         =*/ n_embd,
            /*pos            =*/ pos.data(),
            /*n_seq_id       =*/ n_seq_id.data(),
            /*seq_id         =*/ seq_ids.data(),
            /*logits         =*/ logits.data(),
        };
        for (int i = 0; i < n_tokens; i++) {
            batch.pos     [i] = pos_0 + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }
};

bool llava_eval_image_embed(llama_context * ctx_llama, const struct llava_image_embed * image_embed, int n_batch, int * n_past) {
    int n_embd  = llama_n_embd(llama_get_model(ctx_llama));

    for (int i = 0; i < image_embed->n_image_pos; i += n_batch) {
        int n_eval = image_embed->n_image_pos - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        float * embd = image_embed->embed+i*n_embd;
        llava_embd_batch llava_batch = llava_embd_batch(embd, n_embd, n_eval, *n_past, 0);
        if (llama_decode(ctx_llama, llava_batch.batch)) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

struct llava_image_embed * llava_image_embed_make_with_bytes(struct clip_ctx * ctx_clip, int n_threads, const unsigned char * image_bytes, int image_bytes_length) {
    clip_image_u8 * img = clip_image_u8_init();
    if (!clip_image_load_from_bytes(image_bytes, image_bytes_length, img)) {
        clip_image_u8_free(img);
        LOG_ERR("%s: can't load image from bytes, is it a valid image?", __func__);
        return NULL;
    }

    float* image_embed = NULL;
    int n_image_pos = 0;
    bool image_embed_result = llava_image_embed_make_with_clip_img(ctx_clip, n_threads, img, &image_embed, &n_image_pos);
    if (!image_embed_result) {
        clip_image_u8_free(img);
        LOG_ERR("%s: couldn't embed the image\n", __func__);
        return NULL;
    }

    clip_image_u8_free(img);
    auto result = (llava_image_embed*)malloc(sizeof(llava_image_embed));
    result->embed = image_embed;
    result->n_image_pos = n_image_pos;
    return result;
}

static bool load_file_to_bytes(const char* path, unsigned char** bytesOut, long *sizeOut) {
    auto file = fopen(path, "rb");
    if (file == NULL) {
        LOG_ERR("%s: can't read file %s\n", __func__, path);
        return false;
    }

    fseek(file, 0, SEEK_END);
    auto fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    auto buffer = (unsigned char *)malloc(fileSize); // Allocate memory to hold the file data
    if (buffer == NULL) {
        LOG_ERR("%s: failed to alloc %ld bytes for file %s\n", __func__, fileSize, path);
        perror("Memory allocation error");
        fclose(file);
        return false;
    }
    errno = 0;
    size_t ret = fread(buffer, 1, fileSize, file); // Read the file into the buffer
    if (ferror(file)) {
        LOG_ERR("read error: %s", strerror(errno));
        free(buffer);
        fclose(file);
        return false;
    }
    if (ret != (size_t) fileSize) {
        LOG_ERR("unexpectedly reached end of file");
        free(buffer);
        fclose(file);
        return false;
    }
    fclose(file); // Close the file

    *bytesOut = buffer;
    *sizeOut = fileSize;
    return true;
}

struct llava_image_embed * llava_image_embed_make_with_filename(struct clip_ctx * ctx_clip, int n_threads, const char * image_path) {
    unsigned char* image_bytes;
    long image_bytes_length;
    auto loaded = load_file_to_bytes(image_path, &image_bytes, &image_bytes_length);
    if (!loaded) {
        LOG_ERR("%s: failed to load %s\n", __func__, image_path);
        return NULL;
    }

    llava_image_embed *embed = llava_image_embed_make_with_bytes(ctx_clip, n_threads, image_bytes, image_bytes_length);
    free(image_bytes);

    return embed;
}

void llava_image_embed_free(struct llava_image_embed * embed) {
    free(embed->embed);
    free(embed);
}
