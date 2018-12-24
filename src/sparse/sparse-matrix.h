#ifndef __SBLAS_SPARSE_MATRIX_H__
#define __SBLAS_SPARSE_MATRIX_H__
#pragma once

#include <iostream>
#include <vector>
#include <stdio.h>
#include <string.h>
#include "kernel.h"

typedef int8_t int8;
typedef int16_t int16;
typedef int32_t int32;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;

namespace sblas {

enum SBLAS_TRANSPOSE {
    SblasNoTrans = 0,
    SblasTrans   = 1
};

template<typename PosIndex_t, typename ValIndex_t, typename Value_t, const int32 block_width_shift = 5>
class SparseMatrix {
public:
    SparseMatrix() {}
    SparseMatrix(const ValIndex_t *density_matrix, int32 rows, int32 cols, int32 stride, const Value_t *vals, int32 val_table_size, SBLAS_TRANSPOSE trans = SblasNoTrans) {
        CopyForm(density_matrix, rows, cols, stride, vals, val_table_size, trans);
    }
    ~SparseMatrix() { Destroy(); }

    void Destroy();
    void CopyForm(const ValIndex_t *density_matrix, int32 rows, int32 cols, int32 stride, const Value_t *vals, int32 val_table_size, SBLAS_TRANSPOSE trans = SblasNoTrans);
    void CopyTo(Value_t *density_matrix, int32 stride, SBLAS_TRANSPOSE trans = SblasNoTrans);
    void AddMatMat(Value_t *a, int32 m, int32 lda, Value_t *c, int32 ldc, Value_t alpha, Value_t beta);

    int32 NumRows() const { return rows_; }
    int32 NumCols() const { return cols_; }

    bool operator==(const SparseMatrix<PosIndex_t, ValIndex_t, Value_t, block_width_shift>& oth);
    bool SelfTest();

private:
    std::vector<PosIndex_t> pos_index_;
    std::vector<ValIndex_t> val_index_;
    std::vector<Value_t> val_table_;
    std::vector<std::pair<int32, int32> > block_bounds_;
    std::vector<std::pair<int32, int32> > block_index_bounds_;
    int32 rows_;
    int32 cols_;
};

}

#endif
