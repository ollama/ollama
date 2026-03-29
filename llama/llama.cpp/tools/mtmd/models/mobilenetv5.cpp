#include "models.h"

// Helpers for MobileNetV5 Blocks
// RMS Norm 2D - normalizes over channels for each spatial position
ggml_tensor * clip_graph_mobilenetv5::rms_norm_2d(ggml_tensor * inp, ggml_tensor * weight, float eps) {
    // inp: [W, H, C, B]

    ggml_tensor * cur = ggml_permute(ctx0, inp, 2, 1, 0, 3);
    cur = ggml_cont(ctx0, cur);
    cur = ggml_rms_norm(ctx0, cur, eps);

    if (weight) {
        cur = ggml_mul(ctx0, cur, weight);
    }

    cur = ggml_permute(ctx0, cur, 2, 1, 0, 3);
    cur = ggml_cont(ctx0, cur);

    return cur;
}

// Conv2dSame padding - asymmetric SAME padding like PyTorch/TF
ggml_tensor* clip_graph_mobilenetv5::pad_same_2d(ggml_tensor* inp, int kernel_h, int kernel_w, int stride_h, int stride_w, int dilation_h, int dilation_w) {
    const int64_t ih = inp->ne[1];  // height
    const int64_t iw = inp->ne[0];  // width

    // Calculate output size (ceil division)
    const int64_t oh = (ih + stride_h - 1) / stride_h;
    const int64_t ow = (iw + stride_w - 1) / stride_w;

    // Calculate padding needed
    const int64_t pad_h = std::max((int64_t)0, (oh - 1) * stride_h + (kernel_h - 1) * dilation_h + 1 - ih);
    const int64_t pad_w = std::max((int64_t)0, (ow - 1) * stride_w + (kernel_w - 1) * dilation_w + 1 - iw);

    // Split padding asymmetrically
    const int pad_h_top = pad_h / 2;
    const int pad_h_bottom = pad_h - pad_h_top;
    const int pad_w_left = pad_w / 2;
    const int pad_w_right = pad_w - pad_w_left;

    // Apply padding if needed
    // ggml_pad_ext: (ctx, tensor, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3)
    // For [W, H, C, B]: p0=width, p1=height, p2=channels, p3=batch
    if (pad_h > 0 || pad_w > 0) {
        inp = ggml_pad_ext(ctx0, inp,
            pad_w_left, pad_w_right,     // width padding (dim 0)
            pad_h_top, pad_h_bottom,      // height padding (dim 1)
            0, 0,                         // no channel padding (dim 2)
            0, 0);                        // no batch padding (dim 3)
    }

    return inp;
}


// Edge Residual Block (Stage 0)
ggml_tensor * clip_graph_mobilenetv5::build_edge_residual(ggml_tensor * inp, const mobilenetv5_block & block, int stride) {
    ggml_tensor * cur = inp;

    // 1. Expansion Conv (3x3)
    if (stride == 2) {
        // Case: Downsampling (Block 0)
        // Replicates Conv2dSame(kernel=3, stride=2)
        cur = pad_same_2d(cur, 3, 3, stride, stride);
        cur = ggml_conv_2d_direct(ctx0, block.s0_conv_exp_w, cur, stride, stride, 0, 0, 1, 1);
    } else {
        // Case: Normal 3x3 Block (Block 1, 2)
        // Replicates Conv2d(kernel=3, stride=1, padding=1)
        cur = ggml_conv_2d_direct(ctx0, block.s0_conv_exp_w, cur, stride, stride, 1, 1, 1, 1);
    }

    // BN + Activation
    if (block.s0_bn1_w) cur = rms_norm_2d(cur, block.s0_bn1_w);
    cur = ggml_gelu(ctx0, cur);

    // 2. Pointwise Linear Conv (1x1)
    // 1x1 Convs usually have padding=0 and stride=1
    cur = ggml_conv_2d_direct(ctx0, block.s0_conv_pwl_w, cur, 1, 1, 0, 0, 1, 1);
    if (block.s0_bn2_w) cur = rms_norm_2d(cur, block.s0_bn2_w);

    // 3. Residual Connection
    // Only apply residual if spatial dimensions and channels match (stride 1)
    if (stride == 1 && inp->ne[2] == cur->ne[2] && inp->ne[0] == cur->ne[0]) {
        cur = ggml_add(ctx0, cur, inp);
    }

    return cur;
}

// Universal Inverted Residual Block (Stage 1+)
ggml_tensor * clip_graph_mobilenetv5::build_inverted_residual(ggml_tensor * inp, const mobilenetv5_block & block, int stride) {
    ggml_tensor * cur = inp;

    // 1. Depthwise Start (Optional)
    // NOTE: dw_start always has stride=1 (no downsampling here)
    if (block.dw_start_w) {
        int k = block.dw_start_w->ne[0]; // 3 or 5
        int p = k / 2;
        cur = ggml_conv_2d_dw(ctx0, block.dw_start_w, cur, 1, 1, p, p, 1, 1);
        if (block.dw_start_bn_w) cur = rms_norm_2d(cur, block.dw_start_bn_w);
    }

    // 2. Pointwise Expansion (1x1)
    if (block.pw_exp_w) {
        // Standard 1x1 conv, pad=0, stride=1
        cur = ggml_conv_2d_direct(ctx0, block.pw_exp_w, cur, 1, 1, 0, 0, 1, 1);
        if (block.pw_exp_bn_w) cur = rms_norm_2d(cur, block.pw_exp_bn_w);
        cur = ggml_gelu(ctx0, cur);
    }

    // 3. Depthwise Mid (Optional)
    // NOTE: dw_mid is where downsampling happens (stride=2 for first block of stage)
    if (block.dw_mid_w) {
        int k = block.dw_mid_w->ne[0]; // 3 or 5

        if (stride > 1) {
            // Case: Stride 2 (Downsample) -> Use Asymmetric "Same" Padding
            cur = pad_same_2d(cur, k, k, stride, stride);
            cur = ggml_conv_2d_dw(ctx0, block.dw_mid_w, cur, stride, stride, 0, 0, 1, 1); // pad=0
        } else {
            // Case: Stride 1 -> Use Standard Symmetric Padding
            int p = k / 2;
            cur = ggml_conv_2d_dw(ctx0, block.dw_mid_w, cur, stride, stride, p, p, 1, 1);
        }

        if (block.dw_mid_bn_w) cur = rms_norm_2d(cur, block.dw_mid_bn_w);
        cur = ggml_gelu(ctx0, cur);
    }

    // 4. Pointwise Projection (1x1)
    if (block.pw_proj_w) {
        cur = ggml_conv_2d_direct(ctx0, block.pw_proj_w, cur, 1, 1, 0, 0, 1, 1);
        if (block.pw_proj_bn_w) cur = rms_norm_2d(cur, block.pw_proj_bn_w);
    }

    // Apply Layer Scaling if present
    if (block.layer_scale_w) {
        cur = ggml_mul(ctx0, cur, block.layer_scale_w);
    }

    // 5. Residual Connection
    bool same_spatial = (inp->ne[0] == cur->ne[0]) && (inp->ne[1] == cur->ne[1]);
    bool same_channel = (inp->ne[2] == cur->ne[2]);
    if (same_spatial && same_channel) {
        cur = ggml_add(ctx0, cur, inp);
    }

    return cur;
}

// Attention Block (MQA)
ggml_tensor * clip_graph_mobilenetv5::build_mobilenet_attn(ggml_tensor * inp, const mobilenetv5_block & block) {
    ggml_tensor * cur = inp;

    // Norm
    if (block.attn_norm_w) {
        cur = rms_norm_2d(cur, block.attn_norm_w, 1e-6f);
    }

    // 1. Q Calculation
    ggml_tensor * q = ggml_conv_2d_direct(ctx0, block.attn_q_w, cur, 1, 1, 0, 0, 1, 1);

    // 2. K Calculation (Downsampled)
    // Uses Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640)
    ggml_tensor * k_inp = cur;
    if (block.attn_k_dw_w) {
        int k_size = block.attn_k_dw_w->ne[0];  // Usually 3
        k_inp = pad_same_2d(cur, k_size, k_size, 2, 2);  // Apply SAME padding
        k_inp = ggml_conv_2d_dw(ctx0, block.attn_k_dw_w, k_inp, 2, 2, 0, 0, 1, 1);  // padding=0
        if (block.attn_k_norm_w) {
            k_inp = rms_norm_2d(k_inp, block.attn_k_norm_w, 1e-6f);
        }
    }
    ggml_tensor * k = ggml_conv_2d_direct(ctx0, block.attn_k_w, k_inp, 1, 1, 0, 0, 1, 1);

    // 3. V Calculation (Downsampled)
    // Uses Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640)
    ggml_tensor * v_inp = cur;
    if (block.attn_v_dw_w) {
        int v_size = block.attn_v_dw_w->ne[0];  // Usually 3
        v_inp = pad_same_2d(cur, v_size, v_size, 2, 2);  // Apply SAME padding
        v_inp = ggml_conv_2d_dw(ctx0, block.attn_v_dw_w, v_inp, 2, 2, 0, 0, 1, 1);  // padding=0
        if (block.attn_v_norm_w) {
            v_inp = rms_norm_2d(v_inp, block.attn_v_norm_w, 1e-6f);
        }
    }
    ggml_tensor * v = ggml_conv_2d_direct(ctx0, block.attn_v_w, v_inp, 1, 1, 0, 0, 1, 1);

    const int W = cur->ne[0]; const int H = cur->ne[1]; const int B = cur->ne[3];
    const int D = k->ne[2]; // Head dimension
    const int n_head = q->ne[2] / D;
    const int N = W * H;

    // Process Q: [W, H, D*n_head, B] -> [D, N, n_head, B]
    q = ggml_reshape_3d(ctx0, q, N, D*n_head, B);
    q = ggml_reshape_4d(ctx0, q, N, D, n_head, B);
    q = ggml_permute(ctx0, q, 1, 0, 2, 3); // [D, N, n_head, B]
    q = ggml_cont(ctx0, q);

    const int Wk = k->ne[0]; const int Hk = k->ne[1];
    const int M = Wk * Hk;

    // Process K: [Wk, Hk, D, B] -> [D, M, 1, B]
    k = ggml_reshape_3d(ctx0, k, M, D, B);
    k = ggml_reshape_4d(ctx0, k, M, D, 1, B);
    k = ggml_permute(ctx0, k, 1, 0, 2, 3); // [D, M, 1, B]
    k = ggml_cont(ctx0, k);

    // Process V: [Wk, Hk, D, B] -> [M, D, 1, B]
    v = ggml_reshape_3d(ctx0, v, M, D, B);
    v = ggml_reshape_4d(ctx0, v, M, D, 1, B);
    v = ggml_cont(ctx0, v); // [M, D, 1, B]

    // Multi-Query Attention
    float scale = 1.0f / sqrtf((float)D);

    // Step 1: Compute Q @ K.T
    ggml_tensor * scores = ggml_mul_mat(ctx0, k, q);

    scores = ggml_scale(ctx0, scores, scale);

    scores = ggml_soft_max(ctx0, scores);

    ggml_tensor * kqv = ggml_mul_mat(ctx0, v, scores);

    kqv = ggml_permute(ctx0, kqv, 1, 0, 2, 3);
    kqv = ggml_cont(ctx0, kqv);


    kqv = ggml_reshape_3d(ctx0, kqv, N, D * n_head, B);
    kqv = ggml_reshape_4d(ctx0, kqv, W, H, D * n_head, B);
    kqv = ggml_cont(ctx0, kqv);

    // Output projection
    cur = ggml_conv_2d_direct(ctx0, block.attn_o_w, kqv, 1, 1, 0, 0, 1, 1);

    // Residual & Layer Scale
    if (inp->ne[0] == cur->ne[0] && inp->ne[2] == cur->ne[2]) {
        if (block.layer_scale_w) {
            cur = ggml_mul(ctx0, cur, block.layer_scale_w);
        }
        cur = ggml_add(ctx0, cur, inp);
    }

    return cur;
}

ggml_cgraph * clip_graph_mobilenetv5::build() {
    ggml_tensor * inp = build_inp_raw();

    // 1. Stem - Conv2dSame(3, 64, kernel_size=(3, 3), stride=(2, 2))
    ggml_tensor * cur = pad_same_2d(inp, 3, 3, 2, 2);  // Apply SAME padding

    cur = ggml_conv_2d_direct(ctx0, model.mobilenet_stem_conv_w, cur, 2, 2, 0, 0, 1, 1);  // padding=0
    if (model.mobilenet_stem_conv_b) {
        cur = ggml_add(ctx0, cur, model.mobilenet_stem_conv_b);
    }
    if (model.mobilenet_stem_norm_w) cur = rms_norm_2d(cur, model.mobilenet_stem_norm_w);
    cur = ggml_gelu(ctx0, cur);


    // 2. Blocks
    std::vector<ggml_tensor*> intermediate_features;
    const int total_blocks = model.mobilenet_blocks.size();

    auto is_stage_start = [&](int i) {
        if (i == 0) return true;
        for (int end_idx : model.mobilenet_stage_ends) {
            if (i == end_idx + 1) return true;
        }
        return false;
    };

    auto is_fusion_point = [&](int i) {
        if (model.mobilenet_stage_ends.size() >= 4) {
                if (i == model.mobilenet_stage_ends[2]) return true; // End of Stage 2
                if (i == model.mobilenet_stage_ends[3]) return true; // End of Stage 3
        } else {
            if (i == total_blocks - 1) return true;
        }
        return false;
    };

    for (int i = 0; i < total_blocks; i++) {
        const auto & block = model.mobilenet_blocks[i];
        int stride = is_stage_start(i) ? 2 : 1;

        if (block.s0_conv_exp_w)      cur = build_edge_residual(cur, block, stride);
        else if (block.attn_q_w)      cur = build_mobilenet_attn(cur, block);
        else                          cur = build_inverted_residual(cur, block, stride);

        if (is_fusion_point(i)) {

            intermediate_features.push_back(cur);
        }
    }

    // 3. Multi-Scale Fusion Adapter (MSFA)
    if (!intermediate_features.empty()) {

        // A. Reference Resolution: PyTorch implementation uses inputs[0]
        // We assume intermediate_features[0] is the "High Resolution" target.
        // In MobileNet designs, this is typically the feature map with the smallest stride (e.g. 32x32).
        ggml_tensor* target_feat = intermediate_features[0];
        int high_res_w = target_feat->ne[0];
        int high_res_h = target_feat->ne[1];

        std::vector<ggml_tensor*> resized_feats;

        // B. Resize inputs to match inputs[0] (High Resolution)
        for (auto feat : intermediate_features) {
            int feat_w = feat->ne[0];
            int feat_h = feat->ne[1];

            // PyTorch: if feat_size < high_resolution: interpolate
            if (feat_w < high_res_w || feat_h < high_res_h) {
                // Calculate scale factor.
                // Note: PyTorch 'nearest' works on arbitrary float scales.
                // ggml_upscale generally takes integer factors or target sizes depending on helper.
                // Assuming standard power-of-2 scaling (e.g. 16 -> 32 means scale=2).
                int scale_w = high_res_w / feat_w;
                // int scale_h = high_res_h / feat_h;

                // Safety check for non-integer scaling if strictly replicating
                GGML_ASSERT(high_res_w % feat_w == 0);

                // Upsample (Nearest Neighbor)
                // 2 is the scale factor
                feat = ggml_upscale(ctx0, feat, scale_w, ggml_scale_mode::GGML_SCALE_MODE_NEAREST);
            }
            resized_feats.push_back(feat);
        }

        // C. Concatenate at High Resolution (Channel Dim = 2 in ggml)
        cur = resized_feats[0];
        for (size_t k = 1; k < resized_feats.size(); ++k) {
            cur = ggml_concat(ctx0, cur, resized_feats[k], 2);
        }

        // D. FFN (UniversalInvertedResidual)
        // Structure: Expand Conv -> Norm -> GELU -> Project Conv -> Norm

        // 1. Expansion
        if (model.msfa_ffn_expand_w) {
            // 1x1 Conv
            cur = ggml_conv_2d_direct(ctx0, model.msfa_ffn_expand_w, cur, 1, 1, 0, 0, 1, 1);

            if (model.msfa_ffn_expand_bn) {
                cur = rms_norm_2d(cur, model.msfa_ffn_expand_bn);
            }

            cur = ggml_gelu(ctx0, cur);

        }

        // 2. Projection (No DW because kernel_size=0)
        if (model.msfa_ffn_project_w) {
            // 1x1 Conv
            cur = ggml_conv_2d_direct(ctx0, model.msfa_ffn_project_w, cur, 1, 1, 0, 0, 1, 1);

            // UniversalInvertedResidual typically has a norm after projection
            if (model.msfa_ffn_project_bn) {
                cur = rms_norm_2d(cur, model.msfa_ffn_project_bn);
            }

        }

        // E. Final Downsample to Target Resolution (Output Resolution)
        // PyTorch: matches self.output_resolution (e.g. 16x16)
        const int target_out_res = 16;
        int current_w = cur->ne[0];

        if (current_w > target_out_res) {
            int s = current_w / target_out_res;

            GGML_ASSERT(current_w % target_out_res == 0);

            // Avg Pool: Kernel=s, Stride=s
            cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_AVG, s, s, s, s, 0, 0);

        }

        // F. Final Norm
        if (model.msfa_concat_norm_w) {
            cur = rms_norm_2d(cur, model.msfa_concat_norm_w);

        }
    }

    // 4. Gemma 3n Multimodal Projection (Embedder)
    // Input: 'cur' is [Width, Height, Channels, Batch]
    int W = cur->ne[0];
    int H = cur->ne[1];
    int C = cur->ne[2];
    int B = cur->ne[3];

    GGML_ASSERT(C == hparams.n_embd);

    // 1. Permute and Flatten to [Channels, Tokens, Batch]
    // PyTorch expects (Batch, Seq, Hidden), GGML usually processes (Hidden, Seq, Batch)
    cur = ggml_permute(ctx0, cur, 2, 1, 0, 3); // -> [C, H, W, B]
    cur = ggml_permute(ctx0, cur, 0, 2, 1, 3); // -> [C, W, H, B]
    cur = ggml_cont(ctx0, cur);
    cur = ggml_reshape_3d(ctx0, cur, C, W*H, B);
    cur = ggml_cont(ctx0, cur);


    // 2. FEATURE SCALING
    // PyTorch: vision_outputs *= self.config.vision_config.hidden_size**0.5
    const float scale_factor = sqrtf((float)C);
    cur = ggml_scale(ctx0, cur, scale_factor);


    // 3. SOFT EMBEDDING NORM
    // PyTorch: self._norm(x) * self.weight
    // We must normalize regardless, then multiply if weight exists.
    {
        const float eps = 1e-6f; // Gemma3n uses 1e-6
        cur = ggml_rms_norm(ctx0, cur, eps);

        if (model.mm_soft_emb_norm_w) {
            // Weight shape is (2048,) -> Element-wise broadcast multiply
            cur = ggml_mul(ctx0, cur, model.mm_soft_emb_norm_w);
        }

    }

    // 4. PROJECTION
    // PyTorch: embedding_projection = nn.Linear(vision_hidden, text_hidden, bias=False)
    // Weight stored as [out_features, in_features] = [text_hidden_size, vision_hidden_size]
    if (model.mm_input_proj_w) {
        cur = ggml_mul_mat(ctx0, model.mm_input_proj_w, cur);
    }

    // 5. POST PROJECTION NORM
    // PyTorch: embedding_post_projection_norm = Gemma3nRMSNorm(..., with_scale=False)
    // with_scale=False means weight is registered as buffer with value 1.0
    // So output = rms_norm(x) * 1.0 = rms_norm(x), magnitude ~1
    {
        const float eps = 1e-6f;
        cur = ggml_rms_norm(ctx0, cur, eps);

        if (model.mm_post_proj_norm_w) {
            // If weight is loaded, multiply (should be ~1.0 anyway)
            cur = ggml_mul(ctx0, cur, model.mm_post_proj_norm_w);
        }
    }

    ggml_build_forward_expand(gf, cur);
    return gf;
}
