#pragma once

typedef vector unsigned char vec_t;
typedef __vector_quad acc_t;

template <typename TA>
class tinyBLAS_Q0_PPC {
  public:
    tinyBLAS_Q0_PPC(int64_t k,
                    const TA *A, int64_t lda,
                    const block_q8_0 *B, int64_t ldb,
                    float *C, int64_t ldc,
                    int ith, int nth);

    void matmul(int64_t m, int64_t n);
    void matmul_tiled_q0(int64_t m, int64_t n, int64_t mc, int64_t nc, int64_t kc) {
        vec_t A_pack[mc*kc*2];
        vec_t B_pack[nc*kc*2];
        int comparray[mc*kc];
        constexpr bool is_Ablock_q4 = std::is_same_v<TA, block_q4_0>;
        int64_t ytiles = m / mc;
        int64_t xtiles = n / nc;
        int64_t tiles  = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        if (end > tiles) {
            end = tiles;
        }
        for (int64_t job = start; job < end; ++job) {
            int64_t ii = (job / xtiles) * mc;
            int64_t jj = (job % xtiles) * nc;
            for (int64_t kk = 0; kk < k; kk += kc) {
                if constexpr(is_Ablock_q4) {
                    packNormalInt4_large(A + ii*lda + kk, lda, mc, 4, (int8_t*)A_pack, comparray);
                } else {
                    packNormal_large<int8_t, vector signed char>(A + ii*lda + kk, lda, mc, 8, (int8_t*)A_pack, false, comparray);
                }
                packNormal_large<uint8_t, vector unsigned char>(B + jj*ldb + kk, ldb, nc, 8, (uint8_t*)B_pack, true);
                KERNEL_Q0(ii, jj, mc, nc, kc, kk, A_pack, B_pack, comparray);
            }
        }
    }

  private:
    inline void save_res(int ii, int jj, int idx, vector float* fin_res, int RM=4, int RN=4) {
        for (int I = 0; I < RM; I++) {
            for (int J = 0; J < RN; J++) {
                *((float*)(C+ii+((jj+J)*ldc)+I)) = *((float*)&fin_res[idx+I]+J);
            }
        }
    }

    inline void add_save_res(int ii, int jj, int idx, vector float* fin_res, int RM=4, int RN=4) {
        for (int I = 0; I < RM; I++) {
            for (int J = 0; J < RN; J++) {
                float * c_ptr = (float *)(C+ii+((jj+J)*ldc)+I);
                *c_ptr += *((float*)&fin_res[idx+I]+J);
            }
        }
    }

    template<typename ArrayType>
    inline void compute(acc_t* ACC, int c_idx, int s_idx, ArrayType& comparray, vector float* vs, vector float* fin_res) {
        vector signed int vec_C[4];
        vector float CA[4] = {0};
        vector float res[4] = {0};
        __builtin_mma_disassemble_acc(vec_C, ACC);
        for (int i = 0; i < 4; i++) {
            CA[i] = vec_splats((float)(((double)comparray[c_idx+i]) * -128.0));
            res[i] = vec_add(vec_ctf(vec_C[i], 0), CA[i]);
            fin_res[s_idx+i] = vec_madd(res[i], vs[s_idx+i], fin_res[s_idx+i]);
        }
    }

    inline void process_q4_elements(vector signed char (&c)[2], int* ca) {
        const vector signed char lowMask = vec_splats((signed char)0xF);
        const vector unsigned char v4 = vec_splats((unsigned char)0x4);
        const vector signed char v8 = vec_splats((signed char)0x8);
        vector signed int vsum = {0};
        vector signed int vsum2 = {0};
        c[0] = vec_and(c[1], lowMask);
        c[1] = vec_sr(c[1], v4);
        c[0] = vec_sub(c[0], v8);
        c[1] = vec_sub(c[1], v8);
        vsum = vec_sum4s(c[0], vsum);
        vsum2 = vec_sum4s(c[1], vsum2);
        vsum = vec_add(vsum, vsum2);
        *(ca) = vsum[0] + vsum[1] + vsum[2] + vsum[3];
    }

    template <typename V1, typename V2>
    inline void vector_permute_store(V2 &s1, V2 &s2, V2 &s3, V2 &s4, V1 *vecOffset, bool flip) {
        vector unsigned char swiz1 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
        vector unsigned char swiz2 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
        vector unsigned char swiz3 = {0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
        vector unsigned char swiz4 = {4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};
        V2 t1, t2, t3, t4, t5, t6, t7, t8;
        vector unsigned char xor_vector;
        uint8_t flip_vec = 0x80;
        xor_vector = vec_splats(flip_vec);
        t1 = vec_perm(s1, s2, swiz1);
        t2 = vec_perm(s1, s2, swiz2);
        t3 = vec_perm(s3, s4, swiz1);
        t4 = vec_perm(s3, s4, swiz2);
        t5 = vec_perm(t1, t3, swiz3);
        t6 = vec_perm(t1, t3, swiz4);
        t7 = vec_perm(t2, t4, swiz3);
        t8 = vec_perm(t2, t4, swiz4);
        if (flip == true) {
            t5 = vec_xor(t5, xor_vector);
            t6 = vec_xor(t6, xor_vector);
            t7 = vec_xor(t7, xor_vector);
            t8 = vec_xor(t8, xor_vector);
        }
        vec_xst(t5, 0, vecOffset);
        vec_xst(t6, 0, vecOffset+16);
        vec_xst(t7, 0, vecOffset+32);
        vec_xst(t8, 0, vecOffset+48);
    }

    template<int RM, int RN>
    inline void kernel(int64_t ii, int64_t jj) {
        if constexpr(RM == 4 && RN == 8) {
            KERNEL_4x8(ii,jj);
        } else if constexpr(RM == 8 && RN == 4) {
            KERNEL_8x4(ii,jj);
        } else if constexpr(RM == 8 && RN == 8) {
            KERNEL_8x8(ii,jj);
        } else {
            assert(false && "RN/RM values not supported");
        }
    }
    template<int size>
    void packNormalInt4(const TA* a, int64_t lda, int rows, int cols, int8_t* vec, std::array<int, size>& comparray);
    template<typename VA, typename VB>
    void packNormal(const block_q8_0* a, int64_t lda, int rows, int cols, VA* vec, bool flip);
    void mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n);
    void KERNEL_4x8(int64_t ii, int64_t jj);
    void KERNEL_8x4(int64_t ii, int64_t jj);
    void KERNEL_8x8(int64_t ii, int64_t jj);
    void gemm_small(int64_t m0, int64_t m, int64_t n0, int64_t n, int RM, int RN);
    template <int RM, int RN>
    void gemm(int64_t m0, int64_t m, int64_t n0, int64_t n);

    void compute_scale(int64_t ii, int64_t jj, int blk, vector float* vs){
        for (int I = 0; I<8; I++) {
            float a_scale = unhalf((A+((ii+I)*lda)+blk)->d);
            for (int J = 0; J<4; J++) {
                *((float*)&vs[I]+J) = (a_scale * unhalf((B+((jj+J)*ldb)+blk)->d));
                *((float*)&vs[I+8]+J) = (a_scale * unhalf((B+((jj+J+4)*ldb)+blk)->d));
             }
         }
    }

    inline void process_q8_elements(const int8_t *qs, int *ca) {
        vector signed char c1 = vec_xl(0, qs);
        vector signed char c2 = vec_xl(16, qs);
        vector signed int vsum1 = {0};
        vector signed int vsum2 = {0};
        vsum1 = vec_sum4s(c1, vsum1);
        vsum2 = vec_sum4s(c2, vsum2);
        vector signed int vsum = vec_add(vsum1, vsum2);
        *ca = vsum[0] + vsum[1] + vsum[2] + vsum[3];
    }

    template<typename VA, typename VB>
    void packNormal_large(const block_q8_0* a, int64_t lda, int rows, int cols, VA* vec, bool flip, int* comparray=nullptr) {
        int64_t i, j;
        block_q8_0 *aoffset = NULL;
        VA *vecOffset = NULL;
        block_q8_0* aoffsets[8];
        __vector_pair arr[8];
        VB c[8][2] = {0};
        VB c1[8] = {0}; VB c2[8] = {0};
        aoffset = const_cast<block_q8_0*>(a);
        vecOffset = vec;
        j = (rows >> 3);
        int index = 0;
        if (j > 0) {
            do {
                for (int it = 0; it < 8; it++)
                    aoffsets[it] = aoffset + it*lda;
                aoffset += 8 * lda;
                for (int blk = 0; blk < kc; blk++) {
                    for (int it = 0; it < 8; it++) {
                        arr[it] = __builtin_vsx_lxvp(0, (__vector_pair*)(aoffsets[it]+blk)->qs);
                        __builtin_vsx_disassemble_pair(c[it], &arr[it]);
                        c1[it] = c[it][0];
                        c2[it] = c[it][1];
                        if (comparray){
                            process_q8_elements((aoffsets[it]+ blk)->qs, &comparray[index + 8*blk + it]);
                        }
                    }
                    vector_permute_store<VA, VB>(c1[0], c1[1], c1[2], c1[3], vecOffset, flip);
                    vector_permute_store<VA, VB>(c2[0], c2[1], c2[2], c2[3], vecOffset+64, flip);
                    vector_permute_store<VA, VB>(c1[4], c1[5], c1[6], c1[7], vecOffset+128, flip);
                    vector_permute_store<VA, VB>(c2[4], c2[5], c2[6], c2[7], vecOffset+192, flip);
                    vecOffset += 256;
                }
                j--;
                index += 8*kc;
            } while(j > 0);
        }

    }

    void packNormalInt4_large(const TA* a, int64_t lda, int rows, int cols, int8_t* vec, int*comparray) {
        int64_t i, j;
        TA *aoffset = NULL;
        int8_t *vecOffset = NULL;
        TA *aoffset1 = NULL, *aoffset2 = NULL, *aoffset3 = NULL, *aoffset4 = NULL;
        TA *aoffset5 = NULL, *aoffset6 = NULL, *aoffset7 = NULL, *aoffset8 = NULL;
        vector signed char c1[2] = {0}, c2[2] = {0}, c3[2] = {0}, c4[2] = {0};
        vector signed char c5[2] = {0}, c6[2] = {0}, c7[2] = {0}, c8[2] = {0};
        aoffset = const_cast<TA*>(a);
        vecOffset = vec;
        int index = 0;
        j = (rows >> 3);
        if (j > 0) {
            do {
                aoffset1 = aoffset;
                aoffset2 = aoffset1 + lda;
                aoffset3 = aoffset2 + lda;
                aoffset4 = aoffset3 + lda;
                aoffset5 = aoffset4 + lda;
                aoffset6 = aoffset5 + lda;
                aoffset7 = aoffset6 + lda;
                aoffset8 = aoffset7 + lda;
                aoffset += 8 * lda;
                for (int blk = 0; blk < kc; blk++) {
                    c1[1] = reinterpret_cast<vector signed char>(vec_xl(0, (aoffset1+blk)->qs));
                    c2[1] = reinterpret_cast<vector signed char>(vec_xl(0, (aoffset2+blk)->qs));
                    c3[1] = reinterpret_cast<vector signed char>(vec_xl(0, (aoffset3+blk)->qs));
                    c4[1] = reinterpret_cast<vector signed char>(vec_xl(0, (aoffset4+blk)->qs));
                    c5[1] = reinterpret_cast<vector signed char>(vec_xl(0, (aoffset5+blk)->qs));
                    c6[1] = reinterpret_cast<vector signed char>(vec_xl(0, (aoffset6+blk)->qs));
                    c7[1] = reinterpret_cast<vector signed char>(vec_xl(0, (aoffset7+blk)->qs));
                    c8[1] = reinterpret_cast<vector signed char>(vec_xl(0, (aoffset8+blk)->qs));

                    process_q4_elements(c1, &comparray[index + 8*blk+0]);
                    process_q4_elements(c2, &comparray[index + 8*blk+1]);
                    process_q4_elements(c3, &comparray[index + 8*blk+2]);
                    process_q4_elements(c4, &comparray[index + 8*blk+3]);
                    process_q4_elements(c5, &comparray[index + 8*blk+4]);
                    process_q4_elements(c6, &comparray[index + 8*blk+5]);
                    process_q4_elements(c7, &comparray[index + 8*blk+6]);
                    process_q4_elements(c8, &comparray[index + 8*blk+7]);
                    vector_permute_store<int8_t, vector signed char>(c1[0], c2[0], c3[0], c4[0], vecOffset, false);
                    vector_permute_store<int8_t, vector signed char>(c1[1], c2[1], c3[1], c4[1], vecOffset+64, false);
                    vector_permute_store<int8_t, vector signed char>(c5[0], c6[0], c7[0], c8[0], vecOffset+128, false);
                    vector_permute_store<int8_t, vector signed char>(c5[1], c6[1], c7[1], c8[1], vecOffset+192, false);
                    vecOffset += 256;
                }
                j--;
                index += 8*kc;
            } while (j > 0);
        }
    }

    void KERNEL_Q0(int64_t ii, int64_t jj, int64_t mc, int64_t nc, int64_t kc, int64_t l, vec_t *vec_A, vec_t *vec_B, int *comparray) {
        acc_t acc[8];
        for (int i = 0; i < mc ; i += 8) {
            for (int j = 0; j < nc; j += 8) {
                vector float fin_res[16] = {0};
                vector float vs[16] = {0};
                for (int64_t kk = 0; kk < kc; kk+=2) {
                    for (int x = 0; x < 8; x++) {
                        __builtin_mma_xxsetaccz(&acc[x]);
                    }
                    int A_block_idx = (i/8)*(16*kc) + kk*16;
                    int B_block_idx = (j/8)*(16*kc)+ kk*16;
                    vec_t *A_block = &vec_A[A_block_idx];
                    vec_t *B_block = &vec_B[B_block_idx];
                    for (int x = 0; x < 8; x++) {
                        __builtin_mma_xvi8ger4pp(&acc[0], A_block[x],     B_block[x]);
                        __builtin_mma_xvi8ger4pp(&acc[1], A_block[x + 8], B_block[x]);
                        __builtin_mma_xvi8ger4pp(&acc[2], A_block[x],     B_block[x+8]);
                        __builtin_mma_xvi8ger4pp(&acc[3], A_block[x+8],   B_block[x+8]);
                    }
                    compute_scale(ii+i, jj+j, l+kk, vs);
                    int c_index = (i/8)*(8*kc)+ kk*8;
                    int* c_block = &comparray[c_index];
                    compute(&acc[0], 0,  0,  c_block, vs, fin_res);
                    compute(&acc[1], 4,  4,  c_block, vs, fin_res);
                    compute(&acc[2], 0,  8,  c_block, vs, fin_res);
                    compute(&acc[3], 4, 12,  c_block, vs, fin_res);

                    A_block_idx = (i/8)*(16*kc) + (kk+1)*16;
                    B_block_idx = (j/8)*(16*kc)+ (kk+1)*16;
                    A_block = &vec_A[A_block_idx];
                    B_block = &vec_B[B_block_idx];
                    for (int x = 0; x < 8; x++) {
                        __builtin_mma_xvi8ger4pp(&acc[4], A_block[x],     B_block[x]);
                        __builtin_mma_xvi8ger4pp(&acc[5], A_block[x + 8], B_block[x]);
                        __builtin_mma_xvi8ger4pp(&acc[6], A_block[x],     B_block[x+8]);
                        __builtin_mma_xvi8ger4pp(&acc[7], A_block[x+8],   B_block[x+8]);
                    }
                    compute_scale(ii+i, jj+j, l+kk+1, vs);
                    c_index = (i/8)*(8*kc)+ (kk+1)*8;
                    c_block = &comparray[c_index];
                    compute(&acc[4], 0,  0,  c_block, vs, fin_res);
                    compute(&acc[5], 4,  4,  c_block, vs, fin_res);
                    compute(&acc[6], 0,  8,  c_block, vs, fin_res);
                    compute(&acc[7], 4, 12,  c_block, vs, fin_res);

                }
                if (l == 0) {
                    save_res(ii+i,   jj+j,    0,  fin_res);
                    save_res(ii+i+4, jj+j,    4,  fin_res);
                    save_res(ii+i,   jj+j+4,  8,  fin_res);
                    save_res(ii+i+4, jj+j+4, 12,  fin_res);
                } else {
                    add_save_res(ii+i,   jj+j,    0,  fin_res);
                    add_save_res(ii+i+4, jj+j,    4,  fin_res);
                    add_save_res(ii+i,   jj+j+4,  8,  fin_res);
                    add_save_res(ii+i+4, jj+j+4, 12,  fin_res);
                }
            }
        }
    }

    const TA *const A;
    const block_q8_0 *const B;
    float *C;
    const int64_t k;
    int64_t kc;
    const int64_t lda;
    const int64_t ldb;
    const int64_t ldc;
    const int ith;
    const int nth;
};
