#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include "sparse-matrix.h"

namespace sblas {

template<typename PosIndex_t, typename ValIndex_t, typename Value_t, const int32 block_row_shift, const int32 block_col_shift>
void SparseMatrix<PosIndex_t, ValIndex_t, Value_t, block_row_shift, block_col_shift>::Destroy() {
    pos_index_.clear();
    val_index_.clear();
    val_table_.clear();
    block_bounds_.clear();
    block_index_bounds_.clear();
    rows_ = 0;
    cols_ = 0;
}

template<typename PosIndex_t, typename ValIndex_t, typename Value_t, const int32 block_row_shift, const int32 block_col_shift>
void SparseMatrix<PosIndex_t, ValIndex_t, Value_t, block_row_shift, block_col_shift>::CopyForm(const ValIndex_t *density_matrix, int32 rows, int32 cols, int32 stride, const Value_t *vals, int32 val_table_size, SBLAS_TRANSPOSE trans) {
    Destroy();
    // if PosIndex_t = uint8_t, zero_pad_interval = 255
    const int32 zero_pad_interval = (1<<(sizeof(PosIndex_t)*8)) - 1;
    SBLAS_ASSERT(val_table_size >= 0 && val_table_size<=zero_pad_interval);
    if (val_table_size == 0) return;
    const int32 matrix_block_row_width = block_row_shift ? (1<<block_row_shift) : SblasNoTrans == trans ? rows : cols;
    const int32 matrix_block_col_width = (1<<block_col_shift);
    val_table_.resize(val_table_size + 1);
    memcpy(&val_table_[0], vals, val_table_size * sizeof(Value_t));
    val_table_[val_table_size] = 0;
    if (SblasNoTrans == trans) {
        // Row Major
        for (int32 i=0; i<rows; i+=matrix_block_row_width) {
            for (int32 j=0; j<cols; j+=matrix_block_col_width) {
                int32 prev_index = 0, left_bound = pos_index_.size();
                const ValIndex_t *block_matrix = &density_matrix[i * stride + j];
                const int32 row_width = rows - i >= matrix_block_row_width ? matrix_block_row_width : rows - i;
                const int32 col_width = cols - j >= matrix_block_col_width ? matrix_block_col_width : cols - j;
                for (int32 ii=0; ii<row_width; ii++) {
                    const int32 offset = ii*matrix_block_col_width;
                    const ValIndex_t *prows = &block_matrix[ii*stride];
                    for (int32 jj=0; jj<col_width; jj++) {
                        if (!(prows[jj]>=0 && prows[jj]<val_table_size)) continue;
                        int32 pos = jj + offset - prev_index;
                        while (pos>zero_pad_interval) {
                            // filler zero
                            pos_index_.push_back(zero_pad_interval);
                            val_index_.push_back(val_table_size);
                            pos -= zero_pad_interval;
                        }
                        pos_index_.push_back(pos);
                        val_index_.push_back(prows[jj]);
                        prev_index = jj + offset;
                    }
                }
                if (left_bound != pos_index_.size()) {
                    block_index_bounds_.push_back(std::make_pair(left_bound, pos_index_.size()));
                    block_bounds_.push_back(std::make_pair(i, j));
                }
            }
        }
        rows_ = rows;
        cols_ = cols;
    } else {
        // Col Major
        for (int32 i=0; i<cols; i+=matrix_block_row_width) {
            for (int32 j=0; j<rows; j+=matrix_block_col_width) {
                int32 prev_index = 0, left_bound = pos_index_.size();
                const ValIndex_t *block_matrix = &density_matrix[j * stride + i];
                const int32 col_width = cols - i >= matrix_block_row_width ? matrix_block_row_width : cols - i;
                const int32 row_width = rows - j >= matrix_block_col_width ? matrix_block_col_width : rows - j;
                for (int32 ii=0; ii<col_width; ii++) {
                    const int32 offset = ii*matrix_block_col_width;
                    for (int32 jj=0; jj<row_width; jj++) {
                        ValIndex_t val_id = block_matrix[jj*stride + ii];
                        if (!(val_id>=0 && val_id<val_table_size)) continue;
                        int32 pos = offset + jj - prev_index;
                        while (pos>zero_pad_interval) {
                            // filler zero
                            pos_index_.push_back(zero_pad_interval);
                            val_index_.push_back(val_table_size);
                            pos -= zero_pad_interval;
                        }
                        pos_index_.push_back(pos);
                        val_index_.push_back(val_id);
                        prev_index = offset + jj;
                    }
                }
                if (left_bound != pos_index_.size()) {
                    block_index_bounds_.push_back(std::make_pair(left_bound, pos_index_.size()));
                    block_bounds_.push_back(std::make_pair(i, j));
                }
            }
        }
        rows_ = cols;
        cols_ = rows;
    }
}

template<typename PosIndex_t, typename ValIndex_t, typename Value_t, const int32 block_row_shift, const int32 block_col_shift>
void SparseMatrix<PosIndex_t, ValIndex_t, Value_t, block_row_shift, block_col_shift>::CopyTo(Value_t *density_matrix, int32 stride, SBLAS_TRANSPOSE trans) {
    const int32 align_value = (1<<block_col_shift) - 1;
    const int32 pos_size = pos_index_.size(), val_size = val_index_.size(), valid_table_size = val_table_.size() - 1;
    SBLAS_ASSERT(pos_size == val_size);
    PosIndex_t *ppos = &pos_index_[0];
    ValIndex_t *pval = &val_index_[0];
    if (SblasNoTrans == trans) {
        memset(density_matrix, 0, rows_ * stride * sizeof(Value_t));
        for (int32 i=0; i<block_bounds_.size(); i++) {
            int32 pos_offset = 0;
            const int32 start = block_index_bounds_[i].first;
            const int32 end = block_index_bounds_[i].second;
            Value_t *block_matrix = &density_matrix[block_bounds_[i].first * stride + block_bounds_[i].second];
            for (int32 j=start; j<end; j++) {
                pos_offset += ppos[j];
                if (pval[j] == valid_table_size) continue;
                int32 row = pos_offset >> block_col_shift, col = pos_offset & align_value;
                block_matrix[row*stride + col] = val_table_[pval[j]];
            }
        }
    } else {
        memset(density_matrix, 0, cols_ * stride * sizeof(Value_t));
        for (int32 i=0; i<block_bounds_.size(); i++) {
            int32 pos_offset = 0;
            const int32 start = block_index_bounds_[i].first;
            const int32 end = block_index_bounds_[i].second;
            Value_t *block_matrix = &density_matrix[block_bounds_[i].second * stride + block_bounds_[i].first];
            for (int32 j=start; j<end; j++) {
                pos_offset += ppos[j];
                if (pval[j] == valid_table_size) continue;
                int32 row = pos_offset >> block_col_shift, col = pos_offset & align_value;
                block_matrix[col*stride + row] = val_table_[pval[j]];
            }
        }
    }
}

template<typename PosIndex_t, typename ValIndex_t, typename Value_t, const int32 block_row_shift, const int32 block_col_shift>
void SparseMatrix<PosIndex_t, ValIndex_t, Value_t, block_row_shift, block_col_shift>::AddMatMat(Value_t *a, int32 m, int32 lda, Value_t *c, int32 ldc, Value_t alpha, Value_t beta) {
    const int32 matrix_block_row_width = block_row_shift ? (1<<block_row_shift) : rows_;
    const int32 matrix_block_col_width = (1<<block_col_shift);
    const int32 pos_size = pos_index_.size(), val_size = val_index_.size(), valid_table_size = val_table_.size() - 1;
    SBLAS_ASSERT(pos_size == val_size);
    PosIndex_t *ppos = &pos_index_[0];
    ValIndex_t *pval = &val_index_[0];
    int32 pos_offset = 0;
    const int32 k = rows_, n = cols_;
    if (beta != 1.0) {
        sblas_beta_operation_kernel(c, m, n, ldc, beta);
    }
    if (alpha != 0.0) {
#define TRANS_MULTIPLICATION
#ifdef TRANS_MULTIPLICATION
        const int32 aligned_m = (m + 7) & (~7);
        void *p_tmp = nullptr;
        if(NULL == (Value_t*)SBLAS_MEMALIGN(8, aligned_m*(matrix_block_col_width + matrix_block_row_width)*sizeof(Value_t), &p_tmp)) {
            printf("AddMatMat malloc failed!\n");
            return;
        }
        Value_t *sc = (Value_t*)p_tmp, *sa = sc + aligned_m*matrix_block_col_width;
        int32 prev_row_off = -1;
#endif
        for (int32 i=0; i<block_bounds_.size(); i++) {
            const int32 row_off = block_bounds_[i].first;
            const int32 col_off = block_bounds_[i].second;
            const int32 row_width = rows_ - row_off >= matrix_block_row_width ? matrix_block_row_width : rows_ - row_off;
            const int32 col_width = cols_ - col_off >= matrix_block_col_width ? matrix_block_col_width : cols_ - col_off;
            const int32 start = block_index_bounds_[i].first;
            const int32 end = block_index_bounds_[i].second;
            //std::cout << "row off:" << row_off << ", col off:" << col_off << std::endl;
            //std::cout << "row width:" << row_width << ", col width:" << col_width << std::endl;
            //std::cout << "start:" << start << ", end:" << end << std::endl;
#ifndef TRANS_MULTIPLICATION
            sblas_kernel_operation<PosIndex_t, ValIndex_t, Value_t, block_col_shift>(m, col_width, row_width,
                                    &a[row_off], lda, &c[col_off], ldc,
                                    alpha, &ppos[start], &pval[start], end-start, &val_table_[0], valid_table_size);
        }
#else
            if (prev_row_off != row_off) {
                sblas_trans_kernel(&a[row_off], m, row_width/*k*/, lda, sa, aligned_m/*ldsa*/); // trans a
                prev_row_off = row_off;
            }
            sblas_trans_kernel(&c[col_off], m, col_width/*n*/, ldc, sc, aligned_m/*ldsc*/); // trans c
            // C^T = B^T * A^T
            sblas_kernel_operation_trans_ex<PosIndex_t, ValIndex_t, Value_t, block_col_shift>(m, col_width, row_width,
                                    sa, aligned_m/*ldsa*/, sc, aligned_m/*ldsc*/,
                                    alpha, &ppos[start], &pval[start], end-start, &val_table_[0], valid_table_size);
            sblas_trans_kernel(sc, col_width/*n*/, m, aligned_m/*ldsc*/, &c[col_off], ldc); // trans c
        }
        SBLAS_MEMALIGN_FREE(p_tmp);
#endif
    }
}

// TEST
template<typename PosIndex_t, typename ValIndex_t, typename Value_t, const int32 block_row_shift, const int32 block_col_shift>
bool SparseMatrix<PosIndex_t, ValIndex_t, Value_t, block_row_shift, block_col_shift>::operator==(const SparseMatrix<PosIndex_t, ValIndex_t, Value_t, block_row_shift, block_col_shift>& oth) {
    if (this->pos_index_ != oth.pos_index_) return false;
    if (this->val_index_ != oth.val_index_) return false;
    if (this->val_table_ != oth.val_table_) return false;
    if (this->block_bounds_ != oth.block_bounds_) return false;
    if (this->block_index_bounds_ != oth.block_index_bounds_) return false;
    if (this->rows_ != oth.rows_) return false;
    if (this->cols_ != oth.cols_) return false;
    return true;
}

template<typename PosIndex_t, typename ValIndex_t, typename Value_t, const int32 block_row_shift, const int32 block_col_shift>
bool SparseMatrix<PosIndex_t, ValIndex_t, Value_t, block_row_shift, block_col_shift>::SelfTest() {
    if (true) {
        const int32 rows = 3, cols = 2, stride = 2;
        Value_t test_val_table[8] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8};
        ValIndex_t test_matrix[6] = {0, (ValIndex_t)-1, (ValIndex_t)-1, 3, 7, (ValIndex_t)-1};
        Value_t test_output_matrix[6] = {1,1,1,1,1,1};
        this->CopyForm(test_matrix, rows, cols, stride, test_val_table, 8);
        //if (this->pos_index_ != std::vector<PosIndex_t>{0, 3, 1}) return false;
        //if (this->val_index_ != std::vector<ValIndex_t>{0, 3, 7}) return false;
        this->CopyTo(test_output_matrix, stride);
        if (std::vector<Value_t>(test_output_matrix, test_output_matrix+6) != std::vector<Value_t>{1.1, 0, 0, 4.4, 8.8, 0}) return false;
        Value_t a[3] = {3.1, 5, 7};
        Value_t c[2] = {4, 8};
        this->AddMatMat(a, 1, 3, c, 2, 1.3, 2);
        // [92.513, 28.6]
        if (std::fabs(c[0]-92.513) > 1e-3) return false;
        if (std::fabs(c[1]-44.6) > 1e-3) return false;
    }
    if (true) {
        const int32 rows = 2, cols = 3, stride = 3;
        Value_t test_val_table[8] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8};
        ValIndex_t test_matrix[6] = {0, (ValIndex_t)-1, 7, (ValIndex_t)-1, 3, (ValIndex_t)-1};
        Value_t test_output_matrix[6] = {1,1,1,1,1,1};
        this->CopyForm(test_matrix, rows, cols, stride, test_val_table, 8, SblasTrans);
        //if (this->pos_index_ != std::vector<PosIndex_t>{0, 3, 1}) return false;
        //if (this->val_index_ != std::vector<ValIndex_t>{0, 3, 7}) return false;
        this->CopyTo(test_output_matrix, 2);
        if (std::vector<Value_t>(test_output_matrix, test_output_matrix+6) != std::vector<Value_t>{1.1, 0, 0, 4.4, 8.8, 0}) return false;
        this->CopyTo(test_output_matrix, stride, SblasTrans);
        if (std::vector<Value_t>(test_output_matrix, test_output_matrix+6) != std::vector<Value_t>{1.1, 0, 8.8, 0, 4.4, 0}) return false;
        Value_t a[3] = {3.1, 5, 7};
        Value_t c[2] = {4, 8};
        this->AddMatMat(a, 1, 3, c, 2, 1.3, 2);
        // [92.513, 28.6]
        if (std::fabs(c[0]-92.513) > 1e-3) return false;
        if (std::fabs(c[1]-44.6) > 1e-3) return false;
    }
    if (true) {
        auto gen_matrix_random = [](Value_t *&a, int m, int n) {
            a = (Value_t*) malloc (m*n*sizeof(Value_t));
            for (int i=0; i<m*n; i++) {
                a[i] = Value_t((rand() - RAND_MAX/2));
                if (a[i] > 1000 || a[i] < -1000) a[i] = a[i]/500;
                if (a[i] > 1000 || a[i] < -1000) a[i] = a[i]/500;
                if (a[i] > 1000 || a[i] < -1000) a[i] = a[i]/200;
                if (a[i] > 1000 || a[i] < -1000) a[i] = a[i]/100;
                if (a[i] > 1000 || a[i] < -1000) a[i] = a[i]/100;
            }
        };
        sblas::SparseMatrix<uint8, uint8, Value_t> sparse_matrix;
        const int32 m = 1023, n = 511, stride = 512;
        std::vector<bool> tmp(m*stride*0.75, false);
        tmp.resize(m*stride, true);
        std::random_shuffle(tmp.begin(), tmp.end());
        Value_t *table = nullptr;
        ValIndex_t *index = (ValIndex_t*)malloc(m*stride*sizeof(ValIndex_t));
        Value_t *matrix = (Value_t*)malloc(m*stride*sizeof(Value_t));
        Value_t *matrix_copy = (Value_t*)malloc(m*stride*sizeof(Value_t));
        gen_matrix_random(table, 64, 1);
        for (int i=0; i<tmp.size(); i++) {
            if (!tmp[i]) {
                index[i] = (uint8)-1;
                matrix[i] = 0.0f;
            } else {
                index[i] = rand() % 63;
                matrix[i] = table[index[i]];
            }
        }
        bool success = true;
        if (success) {
            sparse_matrix.CopyForm(index, m, n, stride, table, 63, sblas::SblasNoTrans);
            sparse_matrix.CopyTo(matrix_copy, stride, sblas::SblasNoTrans);
            sparse_matrix.CopyForm(index, m, n, stride, table, 63, sblas::SblasNoTrans);
            sparse_matrix.CopyTo(matrix_copy, stride, sblas::SblasNoTrans);
            for (int i=0; i<m; i++) {
                for (int j=0; j<n; j++) {
                    if (matrix[i*stride + j] != matrix_copy[i*stride + j]) {
                        success = false;
                        break;
                    }
                }
                if (!success) break;
            }
        }
        if (success) {
            sparse_matrix.CopyForm(index, m, n, stride, table, 63, sblas::SblasTrans);
            sparse_matrix.CopyTo(matrix_copy, stride, sblas::SblasTrans);
            for (int i=0; i<m; i++) {
                for (int j=0; j<n; j++) {
                    if (matrix[i*stride + j] != matrix_copy[i*stride + j]) {
                        success = false;
                        break;
                    }
                }
                if (!success) break;
            }
        }
        if (table) free(table), table = nullptr;
        if (index) free(index), index = nullptr;
        if (matrix) free(matrix), matrix = nullptr;
        if (!success) return false;
    }
    return true;
}

template class SparseMatrix<uint8, uint8, float>;

}
