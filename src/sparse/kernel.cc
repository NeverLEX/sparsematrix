#include "kernel.h"
#if defined(HAVE_AVX) || defined(HAVE_AVX2) || defined(HAVE_FMA)
#include <xmmintrin.h>
#include <tmmintrin.h>
#include <immintrin.h>
#endif

template<typename type_t>
void sblas_beta_operation_kernel(type_t* c, int m, int n, int ldc, type_t beta) {
    const int nn = n & (~7);
    for (int i=0; i<m; i++) {
        type_t *cc = &c[i*ldc];
        for (int j=0; j<nn; j+=8) {
            cc[j+0] *= beta;
            cc[j+1] *= beta;
            cc[j+2] *= beta;
            cc[j+3] *= beta;
            cc[j+4] *= beta;
            cc[j+5] *= beta;
            cc[j+6] *= beta;
            cc[j+7] *= beta;
        }
        for (int j=nn; j<n; j++) {
            cc[j] *= beta;
        }
    }
}

template<typename type_t>
void sblas_trans_kernel(type_t* a, int m, int n, int lda, type_t* sa, int ldsa) {
    SBLAS_ASSERT(ldsa >= m);
    type_t *aa0, *aa1, *aa2, *aa3, *sa0, *sa1, *sa2, *sa3;
    type_t *aoffset = a;
    type_t *saoffset = sa;
    int mm = (m>>2);
    while (mm>0) {
        aa0 = aoffset;
        aa1 = aa0 + lda;
        aa2 = aa1 + lda;
        aa3 = aa2 + lda;
        aoffset += 4*lda;
        sa0 = saoffset;
        sa1 = sa0 + ldsa;
        sa2 = sa1 + ldsa;
        sa3 = sa2 + ldsa;
        saoffset += 4;
        int nn = (n>>2);
        while (nn>0) {
            sa0[0] = aa0[0];
            sa1[0] = aa0[1];
            sa2[0] = aa0[2];
            sa3[0] = aa0[3];
            aa0 += 4;

            sa0[1] = aa1[0];
            sa1[1] = aa1[1];
            sa2[1] = aa1[2];
            sa3[1] = aa1[3];
            aa1 += 4;

            sa0[2] = aa2[0];
            sa1[2] = aa2[1];
            sa2[2] = aa2[2];
            sa3[2] = aa2[3];
            aa2 += 4;

            sa0[3] = aa3[0];
            sa1[3] = aa3[1];
            sa2[3] = aa3[2];
            sa3[3] = aa3[3];
            aa3 += 4;

            sa0 += 4*ldsa;
            sa1 += 4*ldsa;
            sa2 += 4*ldsa;
            sa3 += 4*ldsa;
            nn--;
        }
        if (n&2) {
            sa0[0] = aa0[0];
            sa1[0] = aa0[1];
            aa0 += 2;

            sa0[1] = aa1[0];
            sa1[1] = aa1[1];
            aa1 += 2;

            sa0[2] = aa2[0];
            sa1[2] = aa2[1];
            aa2 += 2;

            sa0[3] = aa3[0];
            sa1[3] = aa3[1];
            aa3 += 2;

            sa0 += 2*ldsa;
            sa1 += 2*ldsa;
        }
        if (n&1) {
            sa0[0] = aa0[0];
            sa0[1] = aa1[0];
            sa0[2] = aa2[0];
            sa0[3] = aa3[0];
        }
        mm--;
    }
    if (m&2) {
        aa0 = aoffset;
        aa1 = aa0 + lda;
        aoffset += 2*lda;
        sa0 = saoffset;
        sa1 = sa0 + ldsa;
        sa2 = sa1 + ldsa;
        sa3 = sa2 + ldsa;
        saoffset += 2;
        int nn = (n>>2);
        while (nn>0) {
            sa0[0] = aa0[0];
            sa1[0] = aa0[1];
            sa2[0] = aa0[2];
            sa3[0] = aa0[3];
            aa0 += 4;

            sa0[1] = aa1[0];
            sa1[1] = aa1[1];
            sa2[1] = aa1[2];
            sa3[1] = aa1[3];
            aa1 += 4;

            sa0 += 4*ldsa;
            sa1 += 4*ldsa;
            sa2 += 4*ldsa;
            sa3 += 4*ldsa;
            nn--;
        }
        if (n&2) {
            sa0[0] = aa0[0];
            sa1[0] = aa0[1];
            aa0 += 2;

            sa0[1] = aa1[0];
            sa1[1] = aa1[1];
            aa1 += 2;

            sa0 += 2*ldsa;
            sa1 += 2*ldsa;
        }
        if (n&1) {
            sa0[0] = aa0[0];
            sa0[1] = aa1[0];
        }
    }
    if (m&1) {
        aa0 = aoffset;
        sa0 = saoffset;
        sa1 = sa0 + ldsa;
        sa2 = sa1 + ldsa;
        sa3 = sa2 + ldsa;
        int nn = (n>>2);
        while (nn>0) {
            sa0[0] = aa0[0];
            sa1[0] = aa0[1];
            sa2[0] = aa0[2];
            sa3[0] = aa0[3];
            aa0 += 4;

            sa0 += 4*ldsa;
            sa1 += 4*ldsa;
            sa2 += 4*ldsa;
            sa3 += 4*ldsa;
            nn--;
        }
        if (n&2) {
            sa0[0] = aa0[0];
            sa1[0] = aa0[1];
            aa0 += 2;

            sa0 += 2*ldsa;
            sa1 += 2*ldsa;
        }
        if (n&1) {
            sa0[0] = aa0[0];
        }
    }
}

/*
 * @brief matrix multiplication
 * C = A * B^T
 * C: mxn, A: mxk, B: kxn (sparse)
 * C = A * B^T ==>  C^T = B * A^T,   C^T : nxm ,  A^T: kxm
 *
 * sparse matrix B(block_width_shift = 5, matrix_block_width=32):
 * matrix b = [ 0,   1.1,
 *              2.2, 0 ]
 *      pos = [ 0,     1,
 *              32,    33]   // row * matrix_block_width + col
 * val_table = [1.1, 2.2, 3.3, 4.4, 0]
 * ppos = [1, 31]
 * pval = [0, 1]
 * pos_len = 2
 *
 * @param sa sa = a^T
 * @param sc sc = c^T
 * @param ppos valid value position of matirx B
 * @param pval valid value index of matirx B
 * @param pos_len valid value count
 * @param val_table value table, size = valid_table_size + 1, last element is zero
 * @param valid_table_size value table size
 */
template<typename PosIndex_t, typename ValIndex_t, typename Value_t, const int block_col_shift>
void sblas_kernel_operation(int m, int n, int k, Value_t *a, int lda, Value_t *c, int ldc,
    Value_t alpha, PosIndex_t* ppos, ValIndex_t* pval, int pos_len, Value_t *val_table, int valid_table_size) {
    const int align_value = (1<<block_col_shift) - 1;
    Value_t *aa = a, *cc = c;
    Value_t *aa0, *aa1, *aa2, *aa3, *aa4, *aa5, *aa6, *aa7;
    Value_t *cc0, *cc1, *cc2, *cc3, *cc4, *cc5, *cc6, *cc7;

    void *p_tmp = NULL;
    Value_t *val_list = NULL;
    int *row_list = NULL, *col_list = NULL;
    const int aligned_len = (pos_len + 7) & (~7);
    if (NULL == SBLAS_MEMALIGN(8, aligned_len*(2*sizeof(int) + sizeof(Value_t)), &p_tmp)) {
        printf("sblas kernel operation error!\n");
        return;
    }
    row_list = (int*)p_tmp;
    col_list = row_list + aligned_len;
    val_list = (Value_t*)(col_list + aligned_len);

    int pos_offset = 0, new_len = 0;
    for (int i=0; i<pos_len; i++) {
        pos_offset += ppos[i];
        if (pval[i] >= valid_table_size) continue;
        row_list[new_len] = pos_offset >> block_col_shift;
        col_list[new_len] = pos_offset & align_value;
        val_list[new_len++] = val_table[pval[i]] * alpha;
    }

    int mm = (m>>3);
    while (mm>0) {
        aa0 = aa;
        aa1 = aa0 + lda;
        aa2 = aa1 + lda;
        aa3 = aa2 + lda;
        aa4 = aa3 + lda;
        aa5 = aa4 + lda;
        aa6 = aa5 + lda;
        aa7 = aa6 + lda;
        aa += 8*lda;
        cc0 = cc;
        cc1 = cc0 + ldc;
        cc2 = cc1 + ldc;
        cc3 = cc2 + ldc;
        cc4 = cc3 + ldc;
        cc5 = cc4 + ldc;
        cc6 = cc5 + ldc;
        cc7 = cc6 + ldc;
        cc += 8*ldc;
        for (int i=0; i<new_len; i++) {
            const int &row = row_list[i], &col = col_list[i];
            const Value_t &val = val_list[i];
            cc0[col] += aa0[row] * val;
            cc1[col] += aa1[row] * val;
            cc2[col] += aa2[row] * val;
            cc3[col] += aa3[row] * val;
            cc4[col] += aa4[row] * val;
            cc5[col] += aa5[row] * val;
            cc6[col] += aa6[row] * val;
            cc7[col] += aa7[row] * val;
        }
        mm--;
    }
    if (m&4) {
        aa0 = aa;
        aa1 = aa0 + lda;
        aa2 = aa1 + lda;
        aa3 = aa2 + lda;
        aa += 4*lda;
        cc0 = cc;
        cc1 = cc0 + ldc;
        cc2 = cc1 + ldc;
        cc3 = cc2 + ldc;
        cc += 4*ldc;
        for (int i=0; i<new_len; i++) {
            const int &row = row_list[i], &col = col_list[i];
            const Value_t &val = val_list[i];
            cc0[col] += aa0[row] * val;
            cc1[col] += aa1[row] * val;
            cc2[col] += aa2[row] * val;
            cc3[col] += aa3[row] * val;
        }
    }
    if (m&2) {
        aa0 = aa;
        aa1 = aa0 + lda;
        aa += 2*lda;
        cc0 = cc;
        cc1 = cc0 + ldc;
        cc += 2*ldc;
        for (int i=0; i<new_len; i++) {
            const int &row = row_list[i], &col = col_list[i];
            const Value_t &val = val_list[i];
            cc0[col] += aa0[row] * val;
            cc1[col] += aa1[row] * val;
        }
    }
    if (m&1) {
        aa0 = aa;
        cc0 = cc;
        for (int i=0; i<new_len; i++) {
            const int &row = row_list[i], &col = col_list[i];
            const Value_t &val = val_list[i];
            cc0[col] += aa0[row] * val;
        }
    }
    SBLAS_MEMALIGN_FREE(p_tmp);
}

template<typename PosIndex_t, typename ValIndex_t, typename Value_t, const int block_col_shift>
void sblas_kernel_operation_naive(int m, int n, int k, Value_t *a, int lda, Value_t *c, int ldc,
    Value_t alpha, PosIndex_t* ppos, ValIndex_t* pval, int pos_len, Value_t *val_table, int valid_table_size) {
    const int align_value = (1<<block_col_shift) - 1;
    int pos_offset = 0;
    for (int i=0; i<pos_len; i++) {
        pos_offset += ppos[i];
        if (pval[i] >= valid_table_size) continue;
        int row = pos_offset >> block_col_shift, col = pos_offset & align_value;
        const Value_t val = val_table[pval[i]] * alpha;
        // AddMatMat implementation
        for (int mm=0; mm<m; mm++) {
            c[mm*ldc + col] += a[mm*lda + row] * val;
        }
        // AddMatMat End
    }
}

template<typename PosIndex_t, typename ValIndex_t, typename Value_t, const int block_col_shift>
void sblas_kernel_operation_trans(int m, int n, int k, Value_t *a, int lda, Value_t *c, int ldc,
    Value_t alpha, PosIndex_t* ppos, ValIndex_t* pval, int pos_len, Value_t *val_table, int valid_table_size) {
    // C^T = B * A^T
    int pos_offset = 0;
    const int mm = m&~7;
    const int align_value = (1<<block_col_shift) - 1;
    for (int i=0; i<pos_len; i++) {
        pos_offset += ppos[i];
        if (pval[i] >= valid_table_size) continue;
        int row = pos_offset >> block_col_shift, col = pos_offset & align_value;
        const Value_t val = val_table[pval[i]] * alpha;
        // AddMatMat implementation
        Value_t *cc = &c[col * ldc], *aa = &a[row * lda];
        for (int j=0; j<mm; j+=8) {
            cc[j+0] += aa[j+0] * val;
            cc[j+1] += aa[j+1] * val;
            cc[j+2] += aa[j+2] * val;
            cc[j+3] += aa[j+3] * val;
            cc[j+4] += aa[j+4] * val;
            cc[j+5] += aa[j+5] * val;
            cc[j+6] += aa[j+6] * val;
            cc[j+7] += aa[j+7] * val;
        }
        for (int j=mm; j<m; j++) {
            cc[j] += aa[j] * val;
        }
        // AddMatMat End
    }
}

template<typename Value_t>
void sblas_kernel_mul_naive(int m, Value_t *a, Value_t *c, int ldc, int* col_list, Value_t* val_list, int col_len) {
    const int mm = m & (~7);
    for (int i=0; i<col_len; i++) {
        int col = col_list[i];
        Value_t val = val_list[i];
        // AddMatMat implementation
        Value_t *cc = &c[col * ldc], *aa = a;
#if defined(HAVE_AVX2) && defined(HAVE_FMA)
#if 1
        // 117 * 1023 * 2048 7.5ms
        __asm__ __volatile__ (
            "movq %0, %%r8\n\t"
            "movq %1, %%r9\n\t"
            "movl %2, %%r10d\n\t"
            "shrl $3, %%r10d\n\t" // r10d = mm/8
            "vbroadcastss %3, %%ymm0\n\t"
            ".loop_inner1_0%=:\n\t"
            "cmpl $4, %%r10d\n\t"
            "jl .loop_inner1_1%=\n\t"
            "vmovups  0*4(%%r8), %%ymm1\n\t"
            "vmovups  8*4(%%r8), %%ymm2\n\t"
            "vmovups 16*4(%%r8), %%ymm3\n\t"
            "vmovups 24*4(%%r8), %%ymm4\n\t"
            "vfmadd231ps  0*4(%%r9), %%ymm0, %%ymm1\n\t" // r9 = aa, ymm0 = val
            "vfmadd231ps  8*4(%%r9), %%ymm0, %%ymm2\n\t" // r9 = aa, ymm0 = val
            "vfmadd231ps 16*4(%%r9), %%ymm0, %%ymm3\n\t" // r9 = aa, ymm0 = val
            "vfmadd231ps 24*4(%%r9), %%ymm0, %%ymm4\n\t" // r9 = aa, ymm0 = val
            "vmovups %%ymm0,  0*4(%%r8)\n\t"
            "vmovups %%ymm1,  8*4(%%r8)\n\t"
            "vmovups %%ymm2, 16*4(%%r8)\n\t"
            "vmovups %%ymm3, 24*4(%%r8)\n\t"
            "addq $ 32*4, %%r8\n\t"
            "addq $ 32*4, %%r9\n\t"
            "subl $4, %%r10d\n\t"
            "jnz .loop_inner1_0%=\n\t"
            ".loop_inner1_1%=:\n\t"
            "vmovups (%%r8), %%ymm2\n\t"
            "vfmadd231ps (%%r9), %%ymm0, %%ymm2\n\t" // r9 = aa, ymm0 = val
            "vmovups %%ymm2, (%%r8)\n\t"
            "addq $ 8*4, %%r8\n\t"
            "addq $ 8*4, %%r9\n\t"
            "subl $ 1, %%r10d\n\t"
            "jnz .loop_inner1_1%=\n\t"
            ".loop_inner_end%=:\n\t"
            :"+r"(cc)                               // output
            :"r"(aa),"r"(mm),"rm"(val)              // input
            :"%r8", "%r9", "%r10", "memory"         // clobbered register
        );
#else
        __m256 *mcc = (__m256*)cc, *maa = (__m256*)aa;
        for (int j=0; j<mm; j+=8) {
            *mcc = _mm256_fmadd_ps(*maa, _mm256_broadcast_ss(&val), *mcc);
            maa++; mcc++;
        }
#endif
#elif defined(HAVE_AVX)
#if 1
        // 117 * 1023 * 2048 7.8ms
        __asm__ __volatile__ (
            "movq %0, %%r8\n\t"
            "movq %1, %%r9\n\t"
            "movl %2, %%r10d\n\t"
            "shrl $3, %%r10d\n\t" // r10d = mm/8
            "vbroadcastss %3, %%ymm0\n\t"
            ".loop_inner1_0%=:\n\t"
            "cmpl $4, %%r10d\n\t"
            "jl .loop_inner1_1%=\n\t"
            "vmulps  0*4(%%r9), %%ymm0, %%ymm1\n\t" // r9 = aa, ymm0 = val
            "vmulps  8*4(%%r9), %%ymm0, %%ymm2\n\t" // r9 = aa, ymm0 = val
            "vmulps 16*4(%%r9), %%ymm0, %%ymm3\n\t" // r9 = aa, ymm0 = val
            "vmulps 24*4(%%r9), %%ymm0, %%ymm4\n\t" // r9 = aa, ymm0 = val
            "vaddps  0*4(%%r8), %%ymm1, %%ymm5\n\t"
            "vaddps  8*4(%%r8), %%ymm2, %%ymm6\n\t"
            "vaddps 16*4(%%r8), %%ymm3, %%ymm7\n\t"
            "vaddps 24*4(%%r8), %%ymm4, %%ymm8\n\t"
            "vmovups %%ymm5,  0*4(%%r8)\n\t"
            "vmovups %%ymm6,  8*4(%%r8)\n\t"
            "vmovups %%ymm7, 16*4(%%r8)\n\t"
            "vmovups %%ymm8, 24*4(%%r8)\n\t"
            "addq $ 32*4, %%r8\n\t"
            "addq $ 32*4, %%r9\n\t"
            "subl $4, %%r10d\n\t"
            "jnz .loop_inner1_0%=\n\t"
            ".loop_inner1_1%=:\n\t"
            "vmulps (%%r9), %%ymm0, %%ymm1\n\t" // r9 = aa, ymm0 = val
            "vaddps (%%r8), %%ymm1, %%ymm2\n\t"
            "vmovups %%ymm2, (%%r8)\n\t"
            "addq $ 8*4, %%r8\n\t"
            "addq $ 8*4, %%r9\n\t"
            "subl $ 1, %%r10d\n\t"
            "jnz .loop_inner1_1%=\n\t"
            ".loop_inner_end%=:\n\t"
            :"+r"(cc)                               // output
            :"r"(aa),"r"(mm),"rm"(val)              // input
            :"%r8", "%r9", "%r10", "memory"         // clobbered register
        );
#else
        __m256 *mcc = (__m256*)cc, *maa = (__m256*)aa;
        for (int j=0; j<mm; j+=8) {
            *mcc = _mm256_add_ps(*mcc, _mm256_mul_ps(*maa, _mm256_broadcast_ss(&val)));
            maa++; mcc++;
        }
#endif
/*
#elif (ARM64)

#elif (ARMV7)
*/
#else
        for (int j=0; j<mm; j+=8) {
            cc[j+0] += aa[j+0] * val;
            cc[j+1] += aa[j+1] * val;
            cc[j+2] += aa[j+2] * val;
            cc[j+3] += aa[j+3] * val;
            cc[j+4] += aa[j+4] * val;
            cc[j+5] += aa[j+5] * val;
            cc[j+6] += aa[j+6] * val;
            cc[j+7] += aa[j+7] * val;
        }
#endif
        for (int j=mm; j<m; j++) {
            cc[j] += aa[j] * val;
        }
    }
}

#define SBLAS_KERNEL_MUL_8(m) \
            cc##m[0] += aa[0] * bbx[m]; \
            cc##m[1] += aa[1] * bbx[m]; \
            cc##m[2] += aa[2] * bbx[m]; \
            cc##m[3] += aa[3] * bbx[m]; \
            cc##m[4] += aa[4] * bbx[m]; \
            cc##m[5] += aa[5] * bbx[m]; \
            cc##m[6] += aa[6] * bbx[m]; \
            cc##m[7] += aa[7] * bbx[m]; \
            cc##m += 8;

#define SBLAS_KERNEL_MUL_4(m) \
            cc##m[0] += aa[0] * bbx[m]; \
            cc##m[1] += aa[1] * bbx[m]; \
            cc##m[2] += aa[2] * bbx[m]; \
            cc##m[3] += aa[3] * bbx[m]; \
            cc##m += 4;

#define SBLAS_KERNEL_MUL_2(m) \
            cc##m[0] += aa[0] * bbx[m]; \
            cc##m[1] += aa[1] * bbx[m]; \
            cc##m += 2;

#define SBLAS_KERNEL_MUL_1(m) \
            cc##m[0] += aa[0] * bbx[m]

template<typename Value_t>
void sblas_kernel_mul(int m, Value_t *a, Value_t *c, int ldc, int* col_list, Value_t* val_list, int col_len) {
    Value_t *cc0, *cc1, *cc2, *cc3, *cc4, *cc5, *cc6, *cc7;
    Value_t *bb = val_list, *bbx = bb, *aa = a;
    int *col = col_list;

    int nn = (col_len>>3);
    while (nn>0) {
        bbx = bb;
        bb += 8;
        cc0 = &c[col[0] * ldc];
        cc1 = &c[col[1] * ldc];
        cc2 = &c[col[2] * ldc];
        cc3 = &c[col[3] * ldc];
        cc4 = &c[col[4] * ldc];
        cc5 = &c[col[5] * ldc];
        cc6 = &c[col[6] * ldc];
        cc7 = &c[col[7] * ldc];
        col += 8;
        aa = a;
        int mm = (m>>3);
        while (mm>0) {
            SBLAS_KERNEL_MUL_8(0);
            SBLAS_KERNEL_MUL_8(1);
            SBLAS_KERNEL_MUL_8(2);
            SBLAS_KERNEL_MUL_8(3);
            SBLAS_KERNEL_MUL_8(4);
            SBLAS_KERNEL_MUL_8(5);
            SBLAS_KERNEL_MUL_8(6);
            SBLAS_KERNEL_MUL_8(7);
            aa  += 8;
            mm--;
        }
        if (m&4) {
            SBLAS_KERNEL_MUL_4(0);
            SBLAS_KERNEL_MUL_4(1);
            SBLAS_KERNEL_MUL_4(2);
            SBLAS_KERNEL_MUL_4(3);
            SBLAS_KERNEL_MUL_4(4);
            SBLAS_KERNEL_MUL_4(5);
            SBLAS_KERNEL_MUL_4(6);
            SBLAS_KERNEL_MUL_4(7);
            aa  += 4;
        }
        if (m&2) {
            SBLAS_KERNEL_MUL_2(0);
            SBLAS_KERNEL_MUL_2(1);
            SBLAS_KERNEL_MUL_2(2);
            SBLAS_KERNEL_MUL_2(3);
            SBLAS_KERNEL_MUL_2(4);
            SBLAS_KERNEL_MUL_2(5);
            SBLAS_KERNEL_MUL_2(6);
            SBLAS_KERNEL_MUL_2(7);
            aa  += 2;
        }
        if (m&1) {
            SBLAS_KERNEL_MUL_1(0);
            SBLAS_KERNEL_MUL_1(1);
            SBLAS_KERNEL_MUL_1(2);
            SBLAS_KERNEL_MUL_1(3);
            SBLAS_KERNEL_MUL_1(4);
            SBLAS_KERNEL_MUL_1(5);
            SBLAS_KERNEL_MUL_1(6);
            SBLAS_KERNEL_MUL_1(7);
        }
        nn--;
    }
    if (col_len&4) {
        bbx = bb;
        bb += 4;
        cc0 = &c[col[0] * ldc];
        cc1 = &c[col[1] * ldc];
        cc2 = &c[col[2] * ldc];
        cc3 = &c[col[3] * ldc];
        col += 4;
        aa = a;
        int mm = (m>>3);
        while (mm>0) {
            SBLAS_KERNEL_MUL_8(0);
            SBLAS_KERNEL_MUL_8(1);
            SBLAS_KERNEL_MUL_8(2);
            SBLAS_KERNEL_MUL_8(3);
            aa  += 8;
            mm--;
        }
        if (m&4) {
            SBLAS_KERNEL_MUL_4(0);
            SBLAS_KERNEL_MUL_4(1);
            SBLAS_KERNEL_MUL_4(2);
            SBLAS_KERNEL_MUL_4(3);
            aa  += 4;
        }
        if (m&2) {
            SBLAS_KERNEL_MUL_2(0);
            SBLAS_KERNEL_MUL_2(1);
            SBLAS_KERNEL_MUL_2(2);
            SBLAS_KERNEL_MUL_2(3);
            aa  += 2;
        }
        if (m&1) {
            SBLAS_KERNEL_MUL_1(0);
            SBLAS_KERNEL_MUL_1(1);
            SBLAS_KERNEL_MUL_1(2);
            SBLAS_KERNEL_MUL_1(3);
        }
    }
    if (col_len&2) {
        bbx = bb;
        bb += 2;
        cc0 = &c[col[0] * ldc];
        cc1 = &c[col[1] * ldc];
        col += 2;
        aa = a;
        int mm = (m>>3);
        while (mm>0) {
            SBLAS_KERNEL_MUL_8(0);
            SBLAS_KERNEL_MUL_8(1);
            aa  += 8;
            mm--;
        }
        if (m&4) {
            SBLAS_KERNEL_MUL_4(0);
            SBLAS_KERNEL_MUL_4(1);
            aa  += 4;
        }
        if (m&2) {
            SBLAS_KERNEL_MUL_2(0);
            SBLAS_KERNEL_MUL_2(1);
            aa  += 2;
        }
        if (m&1) {
            SBLAS_KERNEL_MUL_1(0);
            SBLAS_KERNEL_MUL_1(1);
        }
    }
    if (col_len&1) {
        bbx = bb;
        cc0 = &c[col[0] * ldc];
        aa = a;
        int mm = (m>>3);
        while (mm>0) {
            SBLAS_KERNEL_MUL_8(0);
            aa  += 8;
            mm--;
        }
        if (m&4) {
            SBLAS_KERNEL_MUL_4(0);
            aa  += 4;
        }
        if (m&2) {
            SBLAS_KERNEL_MUL_2(0);
            aa  += 2;
        }
        if (m&1) {
            SBLAS_KERNEL_MUL_1(0);
        }
    }
}

template<typename PosIndex_t, typename ValIndex_t, typename Value_t, const int block_col_shift>
void sblas_kernel_operation_trans_ex(int m, int n, int k, Value_t *a, int lda, Value_t *c, int ldc,
        Value_t alpha, PosIndex_t* ppos, ValIndex_t* pval, int pos_len, Value_t *val_table, int valid_table_size) {
    // C^T = B * A^T
    const int align_value = (1<<block_col_shift) - 1;
    int col_list[1<<block_col_shift];
    Value_t val_list[1<<block_col_shift];

    int pos_offset = 0, row_prev = -1, col_len = 0;
    for (int i=0; i<pos_len; i++) {
        pos_offset += ppos[i];
        if (pval[i] >= valid_table_size) continue;
        const int row = pos_offset >> block_col_shift;
        if (row_prev != row) {
            if (row_prev != -1)
                sblas_kernel_mul_naive<Value_t>(m, &a[row_prev*lda], c, ldc, col_list, val_list, col_len);
            // next row
            col_len = 0;
            row_prev = row;
            col_list[col_len] = pos_offset & align_value;
            val_list[col_len++] = val_table[pval[i]] * alpha;
        } else {
            col_list[col_len] = pos_offset & align_value;
            val_list[col_len++] = val_table[pval[i]] * alpha;
        }
    }
    // rest row
    if (row_prev != -1)
        sblas_kernel_mul_naive<Value_t>(m, &a[row_prev*lda], c, ldc, col_list, val_list, col_len);
}

template void sblas_beta_operation_kernel<float>(float* c, int m, int n, int ldc, float beta);
template void sblas_trans_kernel(float* a, int m, int n, int lda, float* sa, int ldsa);
template void sblas_kernel_operation<uint8_t, uint8_t, float, SBLAS_BLOCK_COL_SHIFT>(int m, int n, int k, float *a, int lda, float *c, int ldc,
                                    float alpha, uint8_t* ppos, uint8_t* pval, int pos_len, float *val_table, int valid_table_size);
template void sblas_kernel_operation_naive<uint8_t, uint8_t, float, SBLAS_BLOCK_COL_SHIFT>(int m, int n, int k, float *a, int lda, float *c, int ldc,
                                    float alpha, uint8_t* ppos, uint8_t* pval, int pos_len, float *val_table, int valid_table_size);
template void sblas_kernel_operation_trans<uint8_t, uint8_t, float, SBLAS_BLOCK_COL_SHIFT>(int m, int n, int k, float *a, int lda, float *c, int ldc,
                                    float alpha, uint8_t* ppos, uint8_t* pval, int pos_len, float *val_table, int valid_table_size);
template void sblas_kernel_operation_trans_ex<uint8_t, uint8_t, float, SBLAS_BLOCK_COL_SHIFT>(int m, int n, int k, float *a, int lda, float *c, int ldc,
                                    float alpha, uint8_t* ppos, uint8_t* pval, int pos_len, float *val_table, int valid_table_size);

