#include "kernel.h"

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

template<typename PosIndex_t, typename ValIndex_t, typename Value_t, const int block_width_shift>
void sblas_kernel_operation_naive(int m, int n, int k, Value_t *a, int lda, Value_t *c, int ldc,
    Value_t alpha, PosIndex_t* ppos, ValIndex_t* pval, int pos_len, Value_t *val_table, int valid_table_size) {
    const int align_value = (1<<block_width_shift) - 1;
    int pos_offset = 0;
    for (int i=0; i<pos_len; i++) {
        pos_offset += ppos[i];
        if (pval[i] >= valid_table_size) continue;
        int row = pos_offset >> block_width_shift, col = pos_offset & align_value;
        const Value_t val = val_table[pval[i]] * alpha;
        // AddMatMat implementation
        for (int mm=0; mm<m; mm++) {
            c[mm*ldc + col] += a[mm*lda + row] * val;
        }
        // AddMatMat End
    }
}

template<typename PosIndex_t, typename ValIndex_t, typename Value_t, const int block_width_shift>
void sblas_kernel_operation_trans(int m, int n, int k, Value_t *a, int lda, Value_t *c, int ldc,
    Value_t alpha, PosIndex_t* ppos, ValIndex_t* pval, int pos_len, Value_t *val_table, int valid_table_size) {
    // C^T = B * A^T
    int pos_offset = 0;
    const int mm = m&~7;
    const int align_value = (1<<block_width_shift) - 1;
    for (int i=0; i<pos_len; i++) {
        pos_offset += ppos[i];
        if (pval[i] >= valid_table_size) continue;
        int row = pos_offset >> block_width_shift, col = pos_offset & align_value;
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
template<typename PosIndex_t, typename ValIndex_t, typename Value_t, const int block_width_shift>
void sblas_kernel_operation(int m, int n, int k, Value_t *a, int lda, Value_t *sa, int ldsa, Value_t *c, int ldc, Value_t *sc, int ldsc,
    Value_t alpha, PosIndex_t* ppos, ValIndex_t* pval, int pos_len, Value_t *val_table, int valid_table_size) {
    const int align_value = (1<<block_width_shift) - 1;
    if (sa == nullptr || sc == NULL) {
        sblas_kernel_operation_naive<PosIndex_t, ValIndex_t, Value_t, block_width_shift>(m, n, k, a, lda, c, ldc, alpha, ppos, pval, pos_len, val_table, valid_table_size);
    } else {
        sblas_trans_kernel(a, m, k, lda, sa, ldsa); // trans a
        sblas_trans_kernel(c, m, n, ldc, sc, ldsc); // trans c
        // C^T = B^T * A^T
        sblas_kernel_operation_trans<PosIndex_t, ValIndex_t, Value_t, block_width_shift>(m, n, k, sa, ldsa, sc, ldsc, alpha, ppos, pval, pos_len, val_table, valid_table_size);
        sblas_trans_kernel(sc, n, m, ldsc, c, ldc); // trans c
    }
}

template void sblas_beta_operation_kernel<float>(float* c, int m, int n, int ldc, float beta);
template void sblas_trans_kernel(float* a, int m, int n, int lda, float* sa, int ldsa);
template void sblas_kernel_operation_naive<uint8_t, uint8_t, float, SBLAS_BLOCK_SHIFT>(int m, int n, int k, float *a, int lda, float *c, int ldc,
                                    float alpha, uint8_t* ppos, uint8_t* pval, int pos_len, float *val_table, int valid_table_size);
template void sblas_kernel_operation_trans<uint8_t, uint8_t, float, SBLAS_BLOCK_SHIFT>(int m, int n, int k, float *a, int lda, float *c, int ldc,
                                    float alpha, uint8_t* ppos, uint8_t* pval, int pos_len, float *val_table, int valid_table_size);
template void sblas_kernel_operation<uint8_t, uint8_t, float, SBLAS_BLOCK_SHIFT>(int m, int n, int k, float *a, int lda, float *sa, int ldsa, float *c,
                                    int ldc, float *sc, int ldsc, float alpha, uint8_t* ppos, uint8_t* pval, int pos_len, float *val_table, int valid_table_size);

