#include <iostream>
#include <cmath>
#include "sparse-matrix.h"

namespace sblas {

template<typename PosIndex_t, typename ValIndex_t, typename Value_t>
void SparseMatrix<PosIndex_t, ValIndex_t, Value_t>::Destroy() {
    pos_index_.clear();
    val_index_.clear();
    val_table_.clear();
    rows_ = 0;
    cols_ = 0;
}

template<typename PosIndex_t, typename ValIndex_t, typename Value_t>
void SparseMatrix<PosIndex_t, ValIndex_t, Value_t>::CopyForm(ValIndex_t *density_matrix, int32 rows, int32 cols, int32 stride, Value_t *vals, int32 val_table_size, SBLAS_TRANSPOSE trans) {
    // if PosIndex_t = uint8_t, zero_pad_interval = 255
    const int32 zero_pad_interval = (1<<(sizeof(PosIndex_t)*8)) - 1;
    Destroy();
    SBLAS_ASSERT(val_table_size >= 0);
    if (val_table_size == 0) return;
    val_table_.resize(val_table_size);
    memcpy(&val_table_[0], vals, val_table_size * sizeof(Value_t));
    if (SblasNoTrans == trans) {
        int32 prev_index = 0;
        for (int32 i=0; i<rows; i++) {
            const int32 offset = i*cols;
            ValIndex_t *prows = &density_matrix[i*stride];
            for (int32 j=0; j<cols; j++) {
                if (!(prows[j]>=0 && prows[j]<val_table_size)) continue;
                int32 pos = j + offset - prev_index;
                while (pos>zero_pad_interval) {
                    // filler zero
                    pos_index_.push_back(zero_pad_interval);
                    val_index_.push_back(0);
                    pos -= zero_pad_interval;
                }
                pos_index_.push_back(pos);
                val_index_.push_back(prows[j]);
                prev_index = j + offset;
            }
        }
        rows_ = rows;
        cols_ = cols;
    } else {
        int32 prev_index = 0;
        for (int32 i=0; i<cols; i++) {
            const int32 offset = i*rows;
            for (int32 j=0; j<rows; j++) {
                ValIndex_t val_id = density_matrix[j*stride + i];
                if (!(val_id>=0 && val_id<val_table_size)) continue;
                int32 pos = j + offset - prev_index;
                while (pos>zero_pad_interval) {
                    // filler zero
                    pos_index_.push_back(zero_pad_interval);
                    val_index_.push_back(0);
                    pos -= zero_pad_interval;
                }
                pos_index_.push_back(pos);
                val_index_.push_back(val_id);
                prev_index = j + offset;
            }
        }
        rows_ = cols;
        cols_ = rows;
    }
}

template<typename PosIndex_t, typename ValIndex_t, typename Value_t>
void SparseMatrix<PosIndex_t, ValIndex_t, Value_t>::CopyTo(Value_t *density_matrix, int32 stride, SBLAS_TRANSPOSE trans) {
    const int32 pos_size = pos_index_.size(), val_size = val_index_.size();
    SBLAS_ASSERT(pos_size == val_size);
    PosIndex_t *ppos = &pos_index_[0];
    ValIndex_t *pval = &val_index_[0];
    if (SblasNoTrans == trans) {
        int32 pos_offset = 0;
        memset(density_matrix, 0, rows_ * stride * sizeof(Value_t));
        for (int32 i=0; i<pos_size; i++) {
            int32 cur_pos = pos_offset + ppos[i];
            int32 row = cur_pos / cols_, col = cur_pos % cols_;
            density_matrix[row*stride + col] = val_table_[(uint32)pval[i]];
            pos_offset = cur_pos;
        }
    } else {
        int32 pos_offset = 0;
        memset(density_matrix, 0, cols_ * stride * sizeof(Value_t));
        for (int32 i=0; i<pos_size; i++) {
            int32 cur_pos = pos_offset + ppos[i];
            int32 row = cur_pos / cols_, col = cur_pos % cols_;
            density_matrix[col*stride + row] = val_table_[(uint32)pval[i]];
            pos_offset = cur_pos;
        }
    }
}

template<typename PosIndex_t, typename ValIndex_t, typename Value_t>
void SparseMatrix<PosIndex_t, ValIndex_t, Value_t>::AddMatMat(Value_t *a, int32 m, int32 lda, Value_t *c, int32 ldc, Value_t alpha, Value_t beta) {
    const int32 pos_size = pos_index_.size(), val_size = val_index_.size();
    SBLAS_ASSERT(pos_size == val_size);
    PosIndex_t *ppos = &pos_index_[0];
    ValIndex_t *pval = &val_index_[0];
    int32 pos_offset = 0;
    const int32 k = rows_, n = cols_;
    if (beta != 1.0) {
        for (int32 i=0; i<m; i++) {
            Value_t *cc = &c[i*ldc];
            for (int32 j=0; j<n; j++) {
                cc[j] *= beta;
            }
        }
    }
    if (alpha != 0.0) {
        for (int32 i=0; i<pos_size; i++) {
            const int32 cur_pos = pos_offset + ppos[i];
            const int32 row = cur_pos / cols_, col = cur_pos % cols_;
            const Value_t val = val_table_[(uint32)pval[i]] * alpha;
            // AddMatMat implementation
            for (int32 mm=0; mm<m; mm++) {
                c[mm*ldc + col] += a[mm*lda + row] * val;
            }
            // AddMatMat End
            pos_offset = cur_pos;
        }
    }
}

// TEST
template<typename PosIndex_t, typename ValIndex_t, typename Value_t>
bool SparseMatrix<PosIndex_t, ValIndex_t, Value_t>::SelfTest() {
    {
        const int32 rows = 3, cols = 2, stride = 2;
        Value_t test_val_table[8] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8};
        ValIndex_t test_matrix[6] = {0, (ValIndex_t)-1, (ValIndex_t)-1, 3, 7, (ValIndex_t)-1};
        Value_t test_output_matrix[6] = {1,1,1,1,1,1};
        this->CopyForm(test_matrix, rows, cols, stride, test_val_table, 8);
        if (this->pos_index_ != std::vector<PosIndex_t>{0, 3, 1}) return false;
        if (this->val_index_ != std::vector<ValIndex_t>{0, 3, 7}) return false;
        this->CopyTo(test_output_matrix, stride);
        if (std::vector<Value_t>(test_output_matrix, test_output_matrix+6) != std::vector<Value_t>{1.1, 0, 0, 4.4, 8.8, 0}) return false;
        Value_t a[3] = {3.1, 5, 7};
        Value_t c[2] = {4, 8};
        this->AddMatMat(a, 1, 3, c, 2, 1.3, 2);
        // [92.513, 28.6]
        if (std::fabs(c[0]-92.513) > 1e-3) return false;
        if (std::fabs(c[1]-44.6) > 1e-3) return false;
    }
    {
        const int32 rows = 2, cols = 3, stride = 3;
        Value_t test_val_table[8] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8};
        ValIndex_t test_matrix[6] = {0, (ValIndex_t)-1, 7, (ValIndex_t)-1, 3, (ValIndex_t)-1};
        Value_t test_output_matrix[6] = {1,1,1,1,1,1};
        this->CopyForm(test_matrix, rows, cols, stride, test_val_table, 8, SblasTrans);
        if (this->pos_index_ != std::vector<PosIndex_t>{0, 3, 1}) return false;
        if (this->val_index_ != std::vector<ValIndex_t>{0, 3, 7}) return false;
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
    {
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
        {
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
        if (success) {
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
        if (table) free(table), table = nullptr;
        if (index) free(index), index = nullptr;
        if (matrix) free(matrix), matrix = nullptr;
        if (!success) return false;
    }
    return true;
}

template class SparseMatrix<uint8, uint8, float>;

}
