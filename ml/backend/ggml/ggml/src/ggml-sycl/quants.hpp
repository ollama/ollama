//
// MIT license
// Copyright (C) 2025 Codeplay Software Ltd.
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_QUANTS_HPP
#define GGML_SYCL_QUANTS_HPP

#include "ggml-common.h"
#include "ggml.h"

namespace ggml_sycl_reordered {


// The reordered block moves quants (qs) and  scales(d) to two
// uniform regions of memory that is contiguous in the same tensor.
// What this means is that instead of having:
// [d0, qs0] [d1, qs1] [d2, qs2] ... [dN, qsN]
// We have:
// [qs0, qs1, qs2, ..., qsN]  [d0, d1, d2, ..., dN]
//
// Notes: out-of-bounds qs will run into d values
// Aligment relies on the allocated size of qs

template <ggml_type type> struct block_q_t;


// qk number of weights / quants in a block
// qr number of weights in a byte (described as 'before dequantization')
//    for quantization types that has low and high bits split, qr is calculated with
//    using the lower bits, e.g for Q6 quants QR6 is 2
// qi number of 32 bit integers needed to represent all the quants from a block (`qs` field)
// See ggml-common.h to see how these are calculated
template <> struct block_q_t<GGML_TYPE_Q4_0> {
    struct traits {
        static constexpr uint32_t qk       = QK4_0;
        static constexpr uint32_t qi       = QI4_0;
        static constexpr uint32_t qr       = QR4_0;
        static constexpr uint32_t vdr_mmvq = 2;
    };

    static constexpr int get_block_offset(const int block_index) { return block_index * (traits::qk / traits::qr); }

    static constexpr int get_d_offset(int nrows, int ncols, const int block_index) {
        return (ncols / traits::qr * nrows) + block_index * sizeof(ggml_half);
    }

    static constexpr int block_to_q8_1_ratio() { return traits::qk / QK8_1; }
};

}  // namespace ggml_sycl_reordered

#endif  // GGML_SYCL_QUANTS_HPP
