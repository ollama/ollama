#include "ggml-quants.h"

#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <openvino/core/except.hpp>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/core/parallel.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/core/type/element_type_traits.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/util/attr_types.hpp>
#include <openvino/runtime/tensor.hpp>
#include <string>
#include <vector>

void unpack_32_4(const uint8_t * data, uint8_t * dst) {
    std::fill_n(dst, 16, 0);
    for (int j = 0; j < 16; ++j) {
        uint8_t x = (data[j] & 0x0F);
        uint8_t y = (data[j] >> 4);
        if (j % 2 != 0) {
            x <<= 4;
            y <<= 4;
        }
        dst[j / 2] |= x;
        dst[8 + j / 2] |= y;  // Last 16 weights are in the higher bits
    }
}

// Extracts (weight, scales, zp) from Q4_0 tensors.
// Data layout is: |16 bit scale|32 x 4bit weights|.
void extract_q4_0_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    const uint64_t bytes_per_block = 18;  // 2 bytes scale, 32x0.5 byte weights

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto * zp = static_cast<uint8_t *>(zp_arr.data());

    bool is_scalar_zp = (zp_arr.get_size() == 1);  // Symmetric quantization

    // For Q4_0, zero point is always 8
    if (is_scalar_zp) {
        zp[0] = 8 | (8 << 4);  // Pack two 4-bit values
    }

    ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
        scales[i] = ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block)));
        // For asymmetric quantization, compute per-block zero points
        if (!is_scalar_zp) {
            // Pack two 4-bit zero points per byte
            if (i % 2 == 0) {
                zp[i / 2] = 8;          // Lower nibble
            } else {
                zp[i / 2] |= (8 << 4);  // Upper nibble
            }
        }
        unpack_32_4(data + i * bytes_per_block + 2, weights + i * 16);
    });
}

// Extracts (weight, scales, zp) from Q4_1 tensors.
// Data layout is: |16 bit scale|16 bit min|32 x 4bit weights|.
void extract_q4_1_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       bool use_bias) {
    const uint64_t bytes_per_block = 20;  // 2 bytes scale, 2 bytes min, 32x0.5 byte weights

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    if (use_bias) {
        // Store bias (min) directly as f16 instead of computing u4 zero points
        auto * bias = zp_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
            float scale = static_cast<float>(ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block))));
            float min = static_cast<float>(ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block + 2))));
            scales[i] = ov::float16(scale);
            bias[i] = ov::float16(min);  // bias = min, dequant: w*s + bias
            unpack_32_4(data + i * bytes_per_block + 4, weights + i * 16);
        });
    } else {
        auto * zp = static_cast<uint8_t *>(zp_arr.data());
        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
            float scale = static_cast<float>(ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block))));
            float min = static_cast<float>(ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block + 2))));
            scales[i] = ov::float16(scale);
            // zp = -min / scale (bias = min, so zp = -bias/scale)
            uint8_t zp_val = (scale != 0.0f) ? (uint8_t) std::round(-min / scale) : 0;
            // Pack two 4-bit zero points per byte
            if (i % 2 == 0) {
                zp[i / 2] = zp_val & 0x0F;   // Lower nibble
            } else {
                zp[i / 2] |= (zp_val << 4);  // Upper nibble
            }
            unpack_32_4(data + i * bytes_per_block + 4, weights + i * 16);
        });
    }
}

// Extracts (weight, scales, zp) from Q8_0 tensors.
// Data layout is: |16 bit scale|32 x 8bit weights|.
void extract_q8_0_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    const uint64_t weights_per_block = 32;
    const uint64_t bytes_per_block = 34;  // 2 bytes scale, 32x1 byte weights

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto * zp = static_cast<uint8_t *>(zp_arr.data());

    bool is_scalar_zp = (zp_arr.get_size() == 1);  // Symmetric quantization

    // For Q8_0, zero point is always 128
    if (is_scalar_zp) {
        zp[0] = 128;
    }

    ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
        uint8_t * block_data = data + i * bytes_per_block;
        scales[i] = ov::float16::from_bits(*(uint16_t *) block_data);
        // For asymmetric quantization, store per-block zero points
        if (!is_scalar_zp) {
            zp[i] = 128;
        }
        for (size_t j = 0; j < weights_per_block; ++j) {
            uint8_t x = block_data[j + 2];  // j+2 to skip the scale bytes.
            // Original data is in int8_t, so we add a bias of -128 and invert the first bit.
            x ^= 1 << 7;
            weights[i * weights_per_block + j] = x;
        }
    });
}

void unpack_256_4(const uint8_t * data, uint8_t * dst) {
    // Initialize the output array with zeros
    std::fill_n(dst, 128, 0);

    for (size_t i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            uint8_t x = (data[i * 32 + j] & 0x0F);
            uint8_t y = (data[i * 32 + j] >> 4);
            if (j % 2 != 0) {
                x <<= 4;
                y <<= 4;
            }
            dst[i * 32 + j / 2] |= x;
            dst[i * 32 + 16 + j / 2] |= y;  // Last 16 weights are in the higher bits
        }
    }
}

void extract_q4_k_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       bool use_bias) {
    const uint64_t bytes_per_block = 2 + 2 + 12 + 128;
    const uint64_t n_super_block = tensor->nb[3] / bytes_per_block;

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    // For bias path, zp_arr holds f16 bias values; for zp path, it holds packed u4 zero points
    auto * zp_u4 = use_bias ? nullptr : static_cast<uint8_t *>(zp_arr.data());
    auto * bias_f16 = use_bias ? zp_arr.data<ov::element_type_traits<ov::element::f16>::value_type>() : nullptr;

    ov::parallel_for(n_super_block, [&](size_t i) {
        uint8_t * block_data = data + i * bytes_per_block;

        // Extract scale factors and offsets
        float scale_scales = static_cast<float>(ov::float16::from_bits(*((uint16_t *) block_data)));
        float scale_mins = static_cast<float>(ov::float16::from_bits(*((uint16_t *) block_data + 1)));

        // Extract qs1 and qs2
        uint8_t * qs1 = block_data + 4;

        // Calculate scales
        float scale_vals[8];
        scale_vals[0] = scale_scales * static_cast<float>((*(qs1) & 0b111111));
        scale_vals[1] = scale_scales * static_cast<float>((*(qs1 + 1) & 0b111111));
        scale_vals[2] = scale_scales * static_cast<float>((*(qs1 + 2) & 0b111111));
        scale_vals[3] = scale_scales * static_cast<float>((*(qs1 + 3) & 0b111111));
        scale_vals[4] = scale_scales * static_cast<float>((*(qs1 + 8) & 0b00001111) | ((*(qs1) >> 6) << 4));
        scale_vals[5] = scale_scales * static_cast<float>((*(qs1 + 9) & 0b00001111) | ((*(qs1 + 1) >> 6) << 4));
        scale_vals[6] = scale_scales * static_cast<float>((*(qs1 + 10) & 0b00001111) | ((*(qs1 + 2) >> 6) << 4));
        scale_vals[7] = scale_scales * static_cast<float>((*(qs1 + 11) & 0b00001111) | ((*(qs1 + 3) >> 6) << 4));

        // Calculate min values (bias = -min)
        float min_vals[8];
        min_vals[0] = scale_mins * static_cast<float>((*(qs1 + 4) & 0b111111));
        min_vals[1] = scale_mins * static_cast<float>((*(qs1 + 5) & 0b111111));
        min_vals[2] = scale_mins * static_cast<float>((*(qs1 + 6) & 0b111111));
        min_vals[3] = scale_mins * static_cast<float>((*(qs1 + 7) & 0b111111));
        min_vals[4] = scale_mins * static_cast<float>((*(qs1 + 8) >> 4) | ((*(qs1 + 4) >> 6) << 4));
        min_vals[5] = scale_mins * static_cast<float>((*(qs1 + 9) >> 4) | ((*(qs1 + 5) >> 6) << 4));
        min_vals[6] = scale_mins * static_cast<float>((*(qs1 + 10) >> 4) | ((*(qs1 + 6) >> 6) << 4));
        min_vals[7] = scale_mins * static_cast<float>((*(qs1 + 11) >> 4) | ((*(qs1 + 7) >> 6) << 4));

        // Store scales and compute zero points or bias
        for (int j = 0; j < 8; j++) {
            scales[i * 8 + j] = ov::float16(scale_vals[j]);
            if (use_bias) {
                // Store bias = -min directly as f16, dequant: w*s + bias
                bias_f16[i * 8 + j] = ov::float16(-min_vals[j]);
            } else {
                // zp = min / scale (since bias = -min and zp = -bias/scale)
                uint8_t zp_val = (scale_vals[j] != 0.0f) ? (uint8_t) std::round(min_vals[j] / scale_vals[j]) : 0;
                // Pack two 4-bit zero points per byte
                size_t idx = i * 8 + j;
                if (idx % 2 == 0) {
                    zp_u4[idx / 2] = zp_val & 0x0F;
                } else {
                    zp_u4[idx / 2] |= (zp_val << 4);
                }
            }
        }
        unpack_256_4(block_data + 16, weights + i * 128);
    });
}

void extract_q6_k_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    const uint64_t bytes_per_block = 128 + 64 + 16 + 2;
    const uint64_t n_super_block = tensor->nb[3] / bytes_per_block;

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto * zp = static_cast<uint8_t *>(zp_arr.data());

    bool is_scalar_zp = (zp_arr.get_size() == 1);  // Symmetric quantization

    // For Q6_K, zero point is always 32
    if (is_scalar_zp) {
        zp[0] = 32;
    }

    ov::parallel_for(n_super_block, [&](size_t i) {
        uint8_t * block_data = data + i * bytes_per_block;

        float scale_factor =
            static_cast<float>(ov::float16::from_bits(*((uint16_t *) block_data + 104)));  // (128+64+16)/2

        for (size_t j = 0; j < 16; j++) {
            scales[j + i * 16] =
                ov::float16(scale_factor * static_cast<float>(*((int8_t *) (block_data + 128 + 64 + j))));
            // For asymmetric quantization, store per-block zero points
            if (!is_scalar_zp) {
                zp[j + i * 16] = 32;
            }
        }

        uint8_t * ql = block_data;
        uint8_t * qh = block_data + 128;

        for (int64_t j = 0; j < 32; ++j) {
            weights[i * 256 + j] = (ql[j] & 0xF) | (((qh[j] >> 0) & 3) << 4);
            weights[i * 256 + j + 32] = (ql[32 + j] & 0xF) | (((qh[j] >> 2) & 3) << 4);
            weights[i * 256 + j + 64] = (ql[j] >> 4) | (((qh[j] >> 4) & 3) << 4);
            weights[i * 256 + j + 96] = (ql[32 + j] >> 4) | (((qh[j] >> 6) & 3) << 4);
            weights[i * 256 + j + 128] = (ql[64 + j] & 0xF) | (((qh[32 + j] >> 0) & 3) << 4);
            weights[i * 256 + j + 160] = (ql[96 + j] & 0xF) | (((qh[32 + j] >> 2) & 3) << 4);
            weights[i * 256 + j + 192] = (ql[64 + j] >> 4) | (((qh[32 + j] >> 4) & 3) << 4);
            weights[i * 256 + j + 224] = (ql[96 + j] >> 4) | (((qh[32 + j] >> 6) & 3) << 4);
        }
    });
}

static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

void extract_q5_k_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       bool use_bias) {
    const uint64_t bytes_per_block = 4 + 12 + 32 + 128;
    const uint64_t n_super_block = tensor->nb[3] / bytes_per_block;

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    // For bias path, zp_arr holds f16 bias values; for zp path, it holds u8 zero points
    auto * zp_u8 = use_bias ? nullptr : static_cast<uint8_t *>(zp_arr.data());
    auto * bias_f16 = use_bias ? zp_arr.data<ov::element_type_traits<ov::element::f16>::value_type>() : nullptr;

    ov::parallel_for(n_super_block, [&](size_t i) {
        uint8_t * block_data = data + i * bytes_per_block;

        const float d = static_cast<float>(ov::float16::from_bits(*((uint16_t *) block_data)));
        const float min_factor = static_cast<float>(ov::float16::from_bits(*((uint16_t *) block_data + 1)));

        const uint8_t * scales_data = block_data + 4;   // 12 bytes of scales
        const uint8_t * qh = block_data + 4 + 12;       // 32 bytes of high bits
        const uint8_t * ql = block_data + 4 + 12 + 32;  // 128 bytes of low bits

        int is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;

        // Process 2 blocks in one iteration
        for (int j = 0; j < 256; j += 64) {  // 256 = QK_K, so 4 iterations of 64
            uint8_t sc;
            uint8_t m;

            // Get scale and min for first 32 elements
            get_scale_min_k4(is + 0, scales_data, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min_factor * m;

            // Get scale and min for second 32 elements
            get_scale_min_k4(is + 1, scales_data, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min_factor * m;

            scales[i * 8 + is] = ov::float16(d1);
            scales[i * 8 + is + 1] = ov::float16(d2);
            if (use_bias) {
                // Store bias = -min directly as f16, dequant: w*s + bias
                bias_f16[i * 8 + is] = ov::float16(-m1);
                bias_f16[i * 8 + is + 1] = ov::float16(-m2);
            } else {
                // zp = min / scale (since bias = -min and zp = -bias/scale)
                zp_u8[i * 8 + is] = (d1 != 0.0f) ? (uint8_t) std::round(m1 / d1) : 0;
                zp_u8[i * 8 + is + 1] = (d2 != 0.0f) ? (uint8_t) std::round(m2 / d2) : 0;
            }

            // Extract weights for first 32 elements (matching deq formula exactly)
            for (int l = 0; l < 32; ++l) {
                weights[i * 256 + j + l] = (ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
            }

            // Extract weights for second 32 elements
            for (int l = 0; l < 32; ++l) {
                weights[i * 256 + j + l + 32] = (ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
            }

            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    });
}

// TODO Reorder for make_intX_weights

ov::Output<ov::Node> make_int8_weights(ov::Tensor & weight,
                                       ov::Tensor & scales,
                                       ov::Tensor & zp,
                                       size_t group_size,
                                       bool use_bias) {
    ov::Shape orig_shape = weight.get_shape();

    // Expand dimensions for scales and zp/bias
    auto scale_shape = scales.get_shape();
    auto zp_shape = zp.get_shape();
    bool is_scalar_zp = zp_shape.empty();  // Symmetric quantization

    ov::Shape packed_shape = {orig_shape[0], orig_shape[1] / group_size, group_size};

    if (packed_shape[1] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + 1);
    } else {
        scale_shape.push_back(1);
        scales.set_shape(scale_shape);
        // For symmetric quantization, zp remains scalar (don't resize)
        if (!is_scalar_zp) {
            zp_shape.push_back(1);
            zp.set_shape(zp_shape);
        }
    }

    // Create graph nodes
    auto weights_node = std::make_shared<ov::op::v0::Constant>(ov::element::u8, packed_shape,
                                                               static_cast<uint8_t *>(weight.data()), nullptr);
    weights_node->get_rt_info()["__gguf_tensor_holder"] = weight;
    auto scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);
    auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);

    ov::Output<ov::Node> result;
    if (use_bias && !is_scalar_zp) {
        // Bias path: w * s + b (zp tensor holds f16 bias values)
        auto bias_f16 = std::make_shared<ov::op::v0::Constant>(zp);
        auto w_s = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::AutoBroadcastType::NUMPY);
        result = std::make_shared<ov::op::v1::Add>(w_s, bias_f16, ov::op::AutoBroadcastType::NUMPY);
    } else {
        // Zero point path: (w - zp) * s
        auto zero_point = std::make_shared<ov::op::v0::Constant>(zp);
        float zp_value;
        if (ov::op::util::get_single_value(zero_point, zp_value)) {
            zero_point = ov::op::v0::Constant::create(zero_point->get_element_type(), {}, {zp_value});
        }
        auto zero_point_f16 = std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::f16);
        auto w_zp =
            std::make_shared<ov::op::v1::Subtract>(weights_f16, zero_point_f16, ov::op::AutoBroadcastType::NUMPY);
        result = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);
    }

    if (packed_shape.size() != 2) {
        // If not requantized channel-wise case, reshape back to original shape
        auto final_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);
        result = std::make_shared<ov::op::v1::Reshape>(result, final_shape, false);
    }

    return std::make_shared<ov::op::v0::Convert>(result, ov::element::f32);
}

ov::Output<ov::Node> make_int4_weights(ov::Tensor & weight,
                                       ov::Tensor & scales,
                                       ov::Tensor & zp,
                                       size_t group_size,
                                       bool use_bias) {
    ov::Shape orig_weight_shape = weight.get_shape();

    // Expand dimensions for scales and zp/bias
    ov::Shape scale_shape = scales.get_shape();
    auto zp_shape = zp.get_shape();
    bool is_scalar_zp = zp_shape.empty();  // Symmetric quantization

    // Create INT4 weight tensor
    ov::Shape packed_shape = {orig_weight_shape[0], orig_weight_shape[1] / group_size, group_size};

    if (packed_shape[1] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + 1);
    } else {
        scale_shape.push_back(1);
        scales.set_shape(scale_shape);
        // For symmetric quantization, zp remains scalar (don't resize)
        if (!is_scalar_zp) {
            zp_shape.push_back(1);
            zp.set_shape(zp_shape);
        }
    }

    auto weights_node = std::make_shared<ov::op::v0::Constant>(ov::element::u4, packed_shape,
                                                               static_cast<uint8_t *>(weight.data()), nullptr);
    weights_node->get_rt_info()["__gguf_tensor_holder"] = weight;
    auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);
    auto scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    ov::Output<ov::Node> result;
    if (use_bias && !is_scalar_zp) {
        // Bias path: w * s + b (zp tensor holds f16 bias values)
        auto bias_f16 = std::make_shared<ov::op::v0::Constant>(zp);
        auto w_s = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::AutoBroadcastType::NUMPY);
        result = std::make_shared<ov::op::v1::Add>(w_s, bias_f16, ov::op::AutoBroadcastType::NUMPY);
    } else {
        // Zero point path: (w - zp) * s
        auto zero_points_node = std::make_shared<ov::op::v0::Constant>(zp);
        float zp_value;
        if (ov::op::util::get_single_value(zero_points_node, zp_value)) {
            zero_points_node = ov::op::v0::Constant::create(zero_points_node->get_element_type(), {}, {zp_value});
        }
        auto zero_points_f16 = std::make_shared<ov::op::v0::Convert>(zero_points_node, ov::element::f16);
        auto w_zp =
            std::make_shared<ov::op::v1::Subtract>(weights_f16, zero_points_f16, ov::op::AutoBroadcastType::NUMPY);
        result = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);
    }

    if (packed_shape.size() != 2) {
        // If not requantized channel-wise case, reshape back to original shape
        auto final_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_weight_shape.size()},
                                                                  orig_weight_shape);
        result = std::make_shared<ov::op::v1::Reshape>(result, final_shape, false);
    }

    return std::make_shared<ov::op::v0::Convert>(result, ov::element::f32);
}

// Extract quantized weights from tensor and create weight subgraph
std::shared_ptr<ov::Node> extract_quantized_weights(const ggml_tensor * tensor,
                                                    const void * data,
                                                    ov::Tensor & weights,
                                                    ov::Tensor & scales,
                                                    ov::Tensor & zp,
                                                    bool use_bias) {
    // Create a temporary tensor for extraction functions that read from tensor->data
    ggml_tensor temp_tensor = *tensor;
    temp_tensor.data = const_cast<void *>(data);

    // Determine block size based on tensor type
    int64_t weights_per_block;
    bool is_u4;
    switch (tensor->type) {
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q4_K:
        is_u4 = true;
        weights_per_block = 32;
        break;
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q5_K:
        is_u4 = false;
        weights_per_block = 32;
        break;
    case GGML_TYPE_Q6_K:
        is_u4 = false;
        weights_per_block = 16;
        break;
    default:
        throw std::runtime_error("Unsupported quantized type for extraction: " +
                                 std::string(ggml_type_name(tensor->type)));
    }

    // Extract quantized data
    switch (tensor->type) {
    case GGML_TYPE_Q4_0:
        extract_q4_0_data(&temp_tensor, weights, scales, zp);
        break;
    case GGML_TYPE_Q4_1:
        extract_q4_1_data(&temp_tensor, weights, scales, zp, use_bias);
        break;
    case GGML_TYPE_Q4_K:
        extract_q4_k_data(&temp_tensor, weights, scales, zp, use_bias);
        break;
    case GGML_TYPE_Q8_0:
        extract_q8_0_data(&temp_tensor, weights, scales, zp);
        break;
    case GGML_TYPE_Q6_K:
        extract_q6_k_data(&temp_tensor, weights, scales, zp);
        break;
    case GGML_TYPE_Q5_K:
        extract_q5_k_data(&temp_tensor, weights, scales, zp, use_bias);
        break;
    default:
        throw std::runtime_error("Unsupported quantized type: " + std::string(ggml_type_name(tensor->type)));
    }

    // Create the OpenVINO weight subgraph
    ov::Output<ov::Node> weight_node;
    if (is_u4) {
        weight_node = make_int4_weights(weights, scales, zp, weights_per_block, use_bias);
    } else {
        weight_node = make_int8_weights(weights, scales, zp, weights_per_block, use_bias);
    }

    auto result = weight_node.get_node_shared_ptr();
    result->set_friendly_name(tensor->name);
    return result;
}

// Requantize weights to target format, writing to provided buffers
std::shared_ptr<ov::Node> requantize_to_buffers(const ggml_tensor * tensor,
                                                const void * data,
                                                ExtraQuantType requant_type,
                                                int64_t block_size,
                                                ov::Tensor & weights,
                                                ov::Tensor & scales,
                                                ov::Tensor & zp) {
    int64_t n_elements = ggml_nelements(tensor);

    // First dequantize to F32
    std::vector<float> weights_f32(n_elements);
    ggml_get_type_traits(tensor->type)->to_float(data, weights_f32.data(), n_elements);

    // Handle F16 case - just convert and create constant
    if (requant_type == ExtraQuantType::F16) {
        ggml_get_type_traits(GGML_TYPE_F16)->from_float_ref(weights_f32.data(), weights.data(), n_elements);
        auto result = std::make_shared<ov::op::v0::Constant>(weights);
        result->set_friendly_name(tensor->name);
        return result;
    }

    // Requantize to target quantized format
    bool is_u4 = (requant_type == ExtraQuantType::Q4_0_C || requant_type == ExtraQuantType::Q4_0_128);

    if (is_u4) {
        quantize_q4_0(weights_f32.data(), weights, scales, zp, n_elements, block_size);
    } else if (requant_type == ExtraQuantType::Q8_1_C) {
        quantize_q8_1(weights_f32.data(), weights, scales, zp, n_elements, block_size);
    } else {
        quantize_q8_0(weights_f32.data(), weights, scales, zp, n_elements, block_size);
    }

    // Create the OpenVINO weight subgraph
    ov::Output<ov::Node> weight_node;
    if (is_u4) {
        weight_node = make_int4_weights(weights, scales, zp, block_size);
    } else {
        weight_node = make_int8_weights(weights, scales, zp, block_size);
    }

    auto result = weight_node.get_node_shared_ptr();
    result->set_friendly_name(tensor->name);
    return result;
}

OvWeight process_weight_tensor(const ggml_tensor * tensor, const void * data, void * output_base_ptr, bool use_bias) {
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(data != nullptr);

    OvWeight result;

    // Get 2D shape for weights [rows, cols]
    ov::Shape node_shape = {static_cast<size_t>(tensor->ne[1]), static_cast<size_t>(tensor->ne[0])};

    // Handle F16/F32/BF16 weights
    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        ov::element::Type element_type;
        switch (tensor->type) {
        case GGML_TYPE_F32:
            element_type = ov::element::f32;
            break;
        case GGML_TYPE_F16:
            element_type = ov::element::f16;
            break;
        case GGML_TYPE_BF16:
            element_type = ov::element::bf16;
            break;
        default:
            OPENVINO_THROW("Unexpected tensor type in F16/F32/BF16 path");
        }

        if (output_base_ptr && output_base_ptr != data) {
            // Using external buffer - copy data and create shared-memory constant
            size_t tensor_bytes = ggml_nbytes(tensor);
            memcpy(output_base_ptr, data, tensor_bytes);
            result.weights = ov::Tensor(element_type, node_shape, output_base_ptr);
        } else {
            result.weights = ov::Tensor(element_type, node_shape, data);
        }
        result.weight_node = std::make_shared<ov::op::v0::Constant>(result.weights);
        return result;
    }

    // Handle quantized weights
    if (!ggml_is_quantized(tensor->type)) {
        OPENVINO_THROW("Unsupported weight tensor type: ", ggml_type_name(tensor->type));
    }

    result.layout = ggml_openvino_get_extracted_layout(tensor, use_bias);
    const auto & layout = result.layout;
    if (layout.total_size == 0) {
        OPENVINO_THROW("Unsupported quantized type: ", ggml_type_name(tensor->type));
    }

    if (use_bias) {
        OPENVINO_ASSERT(!layout.is_requant,
                        "use_bias is only used for test-backend-ops, which should not have requantization");
        // bias node will be created on the fly and not use backend buffer
        output_base_ptr = nullptr;
    }

    // F16 requant path - no separate scales/zp needed in result
    if (layout.is_requant && layout.requant_type.has_value() && layout.requant_type.value() == ExtraQuantType::F16) {
        if (output_base_ptr) {
            result.weights = ov::Tensor(ov::element::f16, node_shape,
                                        static_cast<uint8_t *>(output_base_ptr) + layout.weights_offset);
        } else {
            result.weights = ov::Tensor(ov::element::f16, node_shape);
        }
        ov::Tensor dummy_scales, dummy_zp;  // Not used for F16
        result.weight_node =
            requantize_to_buffers(tensor, data, ExtraQuantType::F16, 0, result.weights, dummy_scales, dummy_zp);
        return result;
    }

    // Quantized path (normal extraction or quantized requant)
    // Create weight/scale/zp tensors - shared between both paths
    ov::element::Type weight_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
    ov::Shape scale_shape = {node_shape[0], node_shape[1] / layout.weights_per_block};
    ov::Shape zp_shape = layout.is_symmetric ? ov::Shape{} : scale_shape;

    if (output_base_ptr) {
        uint8_t * buf_base = static_cast<uint8_t *>(output_base_ptr);
        result.weights = ov::Tensor(weight_type, node_shape, buf_base + layout.weights_offset);
        result.scales = ov::Tensor(ov::element::f16, scale_shape, buf_base + layout.scales_offset);
        result.zp = ov::Tensor(weight_type, zp_shape, buf_base + layout.zp_offset);
    } else {
        result.weights = ov::Tensor(weight_type, node_shape);
        result.scales = ov::Tensor(ov::element::f16, scale_shape);
        if (use_bias && !layout.is_symmetric) {
            // bias only has effect for asymmetric quant
            result.zp = ov::Tensor(ov::element::f16, zp_shape);
        } else {
            result.zp = ov::Tensor(weight_type, zp_shape);
        }
    }

    if (layout.is_requant && layout.requant_type.has_value()) {
        result.weight_node = requantize_to_buffers(tensor, data, layout.requant_type.value(), layout.weights_per_block,
                                                   result.weights, result.scales, result.zp);
    } else {
        result.weight_node =
            extract_quantized_weights(tensor, data, result.weights, result.scales, result.zp, use_bias);
    }

    return result;
}

void quantize_q4_0(const float * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk) {
    assert(k % qk == 0);
    const int nb = k / qk;

    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto * zp = static_cast<uint8_t *>(zp_arr.data());
    bool is_scalar_zp = (zp_arr.get_size() == 1);  // Symmetric quantization

    // For Q4_0, zero point is always 8
    if (is_scalar_zp) {
        zp[0] = 8 | (8 << 4);  // Pack two 4-bit values
    }

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;  // absolute max
        float max = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i * qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max = v;
            }
        }

        const float d = max / -8;

        if (d == 0) {
            scales[i] = ov::float16(1.0f);
            // zp is already set to 8 for symmetric, or set per-block for asymmetric
            if (!is_scalar_zp) {
                if (i % 2 == 0) {
                    zp[i / 2] = 8;
                } else {
                    zp[i / 2] |= (8 << 4);
                }
            }
            memset(weights + i * qk / 2, 8 | (8 << 4), qk / 2);
            continue;
        }

        const float id = 1.0f / d;
        scales[i] = ov::float16(d);
        // For asymmetric quantization, store per-block zero points
        if (!is_scalar_zp) {
            if (i % 2 == 0) {
                zp[i / 2] = 8;
            } else {
                zp[i / 2] |= (8 << 4);
            }
        }

        for (int j = 0; j < qk / 2; ++j) {
            const float x0 = x[i * qk + 2 * j] * id;
            const float x1 = x[i * qk + 2 * j + 1] * id;
            const uint8_t xi0 = MIN(15, (int8_t) (x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t) (x1 + 8.5f));
            weights[i * qk / 2 + j] = xi0 | (xi1 << 4);
        }
    }
}

void quantize_q8_0(const float * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk) {
    assert(k % qk == 0);
    const int nb = k / qk;

    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto * zp = static_cast<uint8_t *>(zp_arr.data());
    bool is_scalar_zp = (zp_arr.get_size() == 1);  // Symmetric quantization

    // For Q8_0, zero point is always 128
    if (is_scalar_zp) {
        zp[0] = 128;
    }

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;  // absolute max

        for (int j = 0; j < qk; j++) {
            const float v = x[i * qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
            }
        }

        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        scales[i] = ov::float16(d);
        // For asymmetric quantization, store per-block zero points
        if (!is_scalar_zp) {
            zp[i] = 128;
        }

        for (int j = 0; j < qk; ++j) {
            const float x0 = x[i * qk + j] * id;
            const int8_t xi0 = roundf(x0);
            weights[i * qk + j] = (uint8_t) (xi0 + 128);
        }
    }
}

void quantize_q8_1(const float * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk) {
    assert(k % qk == 0);
    const int nb = k / qk;

    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto * zp = static_cast<uint8_t *>(zp_arr.data());
    for (int i = 0; i < nb; i++) {
        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::lowest();

        for (int j = 0; j < qk; j++) {
            const float v = x[i * qk + j];
            if (v < min) {
                min = v;
            }
            if (v > max) {
                max = v;
            }
        }

        const float d = (max - min) / ((1 << 8) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        scales[i] = ov::float16(d);
        // zp = -min / scale (Q8_1 is asymmetric)
        zp[i] = (d != 0.0f) ? (uint8_t) std::round(-min / d) : 0;

        for (int j = 0; j < qk; ++j) {
            const float x0 = (x[i * qk + j] - min) * id;
            const uint8_t xi0 = roundf(x0);
            weights[i * qk + j] = xi0;
        }
    }
}
