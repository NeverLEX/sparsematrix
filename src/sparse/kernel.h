#ifndef __SBLAS_KERNEL_H__
#define __SBLAS_KERNEL_H__
#pragma once

#include <iostream>
#include <cassert>
#include <stdlib.h>

#if defined(_MSC_VER)
# define WIN32_LEAN_AND_MEAN
# define NOMINMAX
# include <windows.h>
#endif

#ifdef _MSC_VER
#include <stdio.h>
#define unlink _unlink
#else
#include <unistd.h>
#endif

#define SBLAS_ASSERT assert
#define SBLAS_MALLOC malloc
#define SBLAS_FREE   free
#define SBLAS_BLOCK_ROW_SHIFT 0
#define SBLAS_BLOCK_COL_SHIFT 8

#if defined(_MSC_VER)
#  define SBLAS_MEMALIGN(align, size, pp_orig) \
  (*(pp_orig) = _aligned_malloc(size, align))
#  define SBLAS_MEMALIGN_FREE(x) if (x) _aligned_free(x), x = NULL;
#elif defined(__CYGWIN__)
#  define SBLAS_MEMALIGN(align, size, pp_orig) \
  (*(pp_orig) = aligned_alloc(align, size))
#  define SBLAS_MEMALIGN_FREE(x) if (x) free(x), x = NULL;
#else
#  define SBLAS_MEMALIGN(align, size, pp_orig) \
     (!posix_memalign(pp_orig, align, size) ? *(pp_orig) : NULL)
#  define SBLAS_MEMALIGN_FREE(x) if (x) free(x), x = NULL;
#endif

template<typename type_t>
void sblas_beta_operation_kernel(type_t* c, int m, int n, int ldc, type_t beta);

template<typename type_t>
void sblas_trans_kernel(type_t* a, int m, int n, int lda, type_t* sa, int ldsa);

template<typename PosIndex_t, typename ValIndex_t, typename Value_t, const int block_col_shift>
void sblas_kernel_operation(int m, int n, int k, Value_t *a, int lda, Value_t *c, int ldc,
    Value_t alpha, PosIndex_t* ppos, ValIndex_t* pval, int pos_len, Value_t *val_table, int valid_table_size);

template<typename PosIndex_t, typename ValIndex_t, typename Value_t, const int block_col_shift>
void sblas_kernel_operation_naive(int m, int n, int k, Value_t *a, int lda, Value_t *c, int ldc,
    Value_t alpha, PosIndex_t* ppos, ValIndex_t* pval, int pos_len, Value_t *val_table, int valid_table_size);

template<typename PosIndex_t, typename ValIndex_t, typename Value_t, const int block_col_shift>
void sblas_kernel_operation_trans(int m, int n, int k, Value_t *a, int lda, Value_t *c, int ldc,
    Value_t alpha, PosIndex_t* ppos, ValIndex_t* pval, int pos_len, Value_t *val_table, int valid_table_size);

template<typename PosIndex_t, typename ValIndex_t, typename Value_t, const int block_col_shift>
void sblas_kernel_operation_trans_ex(int m, int n, int k, Value_t *a, int lda, Value_t *c, int ldc,
    Value_t alpha, PosIndex_t* ppos, ValIndex_t* pval, int pos_len, Value_t *val_table, int valid_table_size);

#endif
