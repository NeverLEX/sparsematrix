#pragma once
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string>
#include <regex>
#include <vector>
#include <map>
#include <iomanip>
#include "sparse-matrix.h"
#include "../utils/dev-tools.h"

#define SGEMM_ALIGN 0x3ffff
#define ALIGN_PTR(PTR, ALIGN, type)  (type*)(((intptr_t)PTR + ALIGN) & (~ALIGN))

class SgemmFilter {
public:
    bool IsMatch(const std::string & func_name) {
        try {
            if (filters_.size() == 0) return true;
            for (int i=0; i<filters_.size(); i++) {
                if (std::regex_match(func_name, filters_[i])) return filters_flag_[i];
            }
            return false;
        } catch (...) {
            return false;
        }
    }

    void ParseFilter(const char *filterstring) {
        const char *r = filterstring, *e = r + strlen(filterstring);
        while (r<e) {
            while (r<e && uint8_t(*r)<=' ') r++;
            const char *s = r;
            while (r<e && uint8_t(*r)>' ' && *r!=';') r++;
            filters_flag_.push_back(*s!='-');
            if (*s=='-') s++;
            filters_.push_back(std::regex(".*" + std::string(s, r-s) + ".*"));
            if (*r==';') r++;
        }
    }
private:
    std::vector<std::regex> filters_;
    std::vector<bool> filters_flag_;
};

static SgemmFilter g_sgemm_filter;

struct StepList {
public:
    StepList(const char *str) { Parse(str); }
    void Parse(const char *param) {
        begin_ = atoi(param);
        if (const char *sep = strchr(param, ':')) end_ = atoi(sep+1);
        else end_ = begin_;
    }
    inline int Begin() { return begin_; }
    inline int End() { return end_; }
private:
    int begin_;
    int end_;
};

class TimeStatis {
public:
    void SaveTimeInfo(const std::string &func_name, int m, int n, int k, float elapsed_ms) {
        time_infos_[func_name].push_back(TimeInfo(m,n,k,elapsed_ms));
        if (func_names_.size() == 0 || func_names_.back() != func_name) func_names_.push_back(func_name);
    }

    void PrintStatis() {
        for (int i=0; i<func_names_.size(); i++) {
            const std::vector<TimeInfo>& ti = time_infos_[func_names_[i]];
            if (i==0) {
                std::cout << "| |";
                for (int j=0; j<ti.size(); j++) {
                    std::cout << " " << ti[j].m << "x" << ti[j].n << "x" << ti[j].k <<  " |";
                }
                std::cout << std::endl;
            }
            std::cout << "| " << func_names_[i] << " |";
            for (int j=0; j<ti.size(); j++) {
                std::cout << " " << ti[j].elapsed_ms <<  "ms |";
            }
            std::cout << std::endl;
        }
    }
private:
    struct TimeInfo {
        int m, n, k;
        float elapsed_ms;
        TimeInfo (int m, int n, int k, float elapsed_ms) : m(m), n(n), k(k), elapsed_ms(elapsed_ms) {}
    };

    std::map<std::string, std::vector<TimeInfo>> time_infos_;
    std::vector<std::string> func_names_;
};

static bool g_check_matrix = true;
static TimeStatis g_time_statis;

/*
 *void cblas_sgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
 *                 OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
 *                 OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda,
 *                 OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);
 *
 * C = alpha * A * B^T + beta * C
 * C : m x n
 * A : m x k
 * B : n x k, B^T : k x n
 *
 */
void cblas_sgemm_baseline(int m, int n, int k, float *a, float *b, float *c, float alpha, float beta);

namespace blas_common {

template<typename type_t>
void gen_matrix_random(type_t *&a, int m, int n) {
    a = (type_t*) malloc (m*n*sizeof(type_t) + SGEMM_ALIGN);
    for (int i=0; i<m*n; i++) {
        a[i] = type_t((rand() - RAND_MAX/2));
        if (a[i] > 1000 || a[i] < -1000) a[i] = a[i]/500;
        if (a[i] > 1000 || a[i] < -1000) a[i] = a[i]/500;
        if (a[i] > 1000 || a[i] < -1000) a[i] = a[i]/200;
        if (a[i] > 1000 || a[i] < -1000) a[i] = a[i]/100;
        if (a[i] > 1000 || a[i] < -1000) a[i] = a[i]/100;
    }
}

template<typename type_t, typename index_t>
void gen_sparse_matrix_random(int m, int n, int stride, float percent, sblas::SparseMatrix<index_t, index_t, type_t> &sparse_matrix) {
    std::vector<bool> tmp(m*stride*percent, false);
    tmp.resize(m*stride, true);
    std::random_shuffle(tmp.begin(), tmp.end());
    type_t *table = nullptr;
    uint8 *index = nullptr;
    gen_matrix_random(table, 256, 1);
    gen_matrix_random(index, m, n);
    for (int i=0; i<tmp.size(); i++) {
        if (!tmp[i]) index[i] = (uint8)-1;
        else index[i] = rand() % 255;
    }
    sparse_matrix.CopyForm(index, m, n, stride, table, 255, sblas::SblasTrans);
    if (table) free(table), table = nullptr;
}

template<typename type_t>
void PrintMatirx(type_t *a, int m, int n, int stride) {
    for (int i=0; i<m; i++) {
        std::cout << (i==0 ? "[" : " ");
        for (int j=0; j<n; j++) {
            std::cout << a[i*stride + j] << (i==m-1 && j==n-1 ? "" : ", ");
        }
        std::cout << (i==m-1 ? "]" : "") << std::endl;
    }
}

template<typename result_t>
void sgemm_check(const char * func_name, result_t *c, result_t *check, int size) {
    if (typeid(*c) == typeid(float) || typeid(*c) == typeid(double)) {
        int diff_count = 0;
        for (int i=0; i<size; i++) {
            const float zz = (c[i]==0) ? 1e-6 : c[i];
            const float xx = (check[i] - c[i])/zz;
            if (xx<-1e-1 || xx > 1e-1) {
                if (diff_count++ > (size)/1e4) {
                    std::cout << "check " << func_name << " result failed, [" << i << "] c:" << c[i] << ", check:" << check[i]
                        << ", diff:" << xx << ", diff_count:" << diff_count << std::endl;
                    return;
                }
            }
        }
    } else {
        for (int i=0; i<size; i++) {
            if (c[i] == check[i]) continue;
            std::cout << "check " << func_name << " result failed, [" << i << "] c:" << c[i] << ", check:" << check[i] << std::endl;
            return;
        }
    }
}

template<typename type_t, typename result_t>
void sgemm_random_invoker(int m, int n, int k, float alpha, float beta,
    void (* sgemm_func)(int m, int n, int k, type_t *a, type_t *b, result_t *c, float alpha, float beta),
    const char *func_name) {
    const int lda = k, ldb = k, ldc = n;
    type_t *a = nullptr, *b = nullptr;
    result_t *c = nullptr, *check = nullptr;
    gen_matrix_random(a, m, lda);
    gen_matrix_random(b, n, ldb);
    gen_matrix_random(c, m, ldc);
    if (g_check_matrix) {
        gen_matrix_random(check, m, ldc);
        for (int i=0; i<m*ldc; i++) check[i] = c[i];
    }
    utility::Timer tm;
    //for (int i=0; i<10; i++) {
        sgemm_func(m, n, k, a, b, c, alpha, beta);
    //}
    g_time_statis.SaveTimeInfo(func_name, m, n, k, tm.elapsed_ms());
    if (g_check_matrix) {
        cblas_sgemm_baseline(m, n, k, a, b, check, alpha, beta);
        sgemm_check(func_name, c, check, m*n);
        free(check), check = nullptr;
    }
    // release
    free(a), a= nullptr;
    free(b), b= nullptr;
    free(c), c= nullptr;
    if (check) free(check), check= nullptr;
}

template<typename type_t, typename result_t>
void sgemm_random_invoker_sparse(int m, int n, int k, float alpha, float beta, const char *func_name) {
    const int lda = k, ldb = k, ldc = n;
    type_t *a = nullptr, *b = nullptr;
    result_t *c = nullptr, *check = nullptr;
    gen_matrix_random(a, m, lda);
    gen_matrix_random(b, n, ldb);
    gen_matrix_random(c, m, ldc);
    sblas::SparseMatrix<uint8, uint8, type_t> sparse_matrix_b;
    gen_sparse_matrix_random<type_t, uint8>(n, ldb, ldb, 0.75f, sparse_matrix_b);
    if (g_check_matrix) {
        sparse_matrix_b.CopyTo(b, ldb, sblas::SblasTrans);
        gen_matrix_random(check, m, ldc);
        for (int i=0; i<m*ldc; i++) check[i] = c[i];
    }
    //PrintMatirx(a, m, k, lda);
    //PrintMatirx(b, n, k, ldb);
    utility::Timer tm;
    //for (int i=0; i<10; i++) {
        //PrintMatirx(c, m, n, ldc);
        sparse_matrix_b.AddMatMat(a, m, k/*lda*/, c, n/*ldc*/, alpha, beta);
        //PrintMatirx(c, m, n, ldc);
        //sgemm_func(m, n, k, a, b, c, alpha, beta);
    //}
    g_time_statis.SaveTimeInfo(func_name, m, n, k, tm.elapsed_ms());
    if (g_check_matrix) {
        //PrintMatirx(check, m, n, ldc);
        cblas_sgemm_baseline(m, n, k, a, b, check, alpha, beta);
        //PrintMatirx(check, m, n, ldc);
        sgemm_check(func_name, c, check, m*n);
        free(check), check = nullptr;
    }
    // release
    free(a), a= nullptr;
    free(b), b= nullptr;
    free(c), c= nullptr;
    if (check) free(check), check= nullptr;
}

template<typename type_t, typename result_t>
void sgemm_random_invoker_ex(int m, int n, int k, float alpha, float beta,
    void (* sgemm_pre_func)(int m, int k, type_t *a, result_t *c),
    void (* sgemm_func)(int m, int n, int k, type_t *a, type_t *b, result_t *c, float alpha, float beta),
    const char *func_name) {
    const int lda = k, ldb = k, ldc = n;
    type_t *a = nullptr, *b = nullptr, *b_precopy = nullptr, *b_precopy_align = nullptr;
    result_t *c = nullptr, *check = nullptr;
    gen_matrix_random(a, m, lda);
    gen_matrix_random(b, n, ldb);
    gen_matrix_random(c, m, ldc);
    if (g_check_matrix) {
        gen_matrix_random(check, m, ldc);
        for (int i=0; i<m*ldc; i++) check[i] = c[i];
    }
    if (sgemm_pre_func) {
        gen_matrix_random(b_precopy, n, ldb);
        b_precopy_align = ALIGN_PTR(b_precopy, SGEMM_ALIGN, type_t);
        sgemm_pre_func(n, k, b, b_precopy_align);
    }
    utility::Timer tm;
    //for (int i=0; i<10; i++) {
        sgemm_func(m, n, k, a, b_precopy_align ? b_precopy_align : b, c, alpha, beta);
    //}
    g_time_statis.SaveTimeInfo(func_name, m, n, k, tm.elapsed_ms());
    if (g_check_matrix) {
        cblas_sgemm_baseline(m, n, k, a, b, check, alpha, beta);
        sgemm_check(func_name, c, check, m*n);
        free(check), check = nullptr;
    }
    // release
    free(a), a= nullptr;
    free(b), b= nullptr;
    free(c), c= nullptr;
    if (check) free(check), check= nullptr;
    if (b_precopy) free(b_precopy), b_precopy= nullptr;
}

#define __LINK_MULTIPLE__(x,y,z) x##_##y##_##z
#define LINK_MULTIPLE(x,y,z) __LINK_MULTIPLE__(x,y,z)
#define __TO_STRING__(x) #x
#define TO_STRING(x) __TO_STRING__(x)

#define SGEMM_INVOKER(func) \
    if (g_sgemm_filter.IsMatch(#func)) { \
        for (int i=arg_m.Begin(); i<=arg_m.End(); i<<=1) { \
            for (int j=arg_n.Begin(); j<=arg_n.End(); j<<=1) { \
                for (int k=arg_k.Begin(); k<=arg_k.End(); k<<=1) { \
                    blas_common::sgemm_random_invoker(i, j, k, 1.8f, 1.2f, func, #func); \
                } \
            } \
        } \
    }

#define SGEMM_SPARSE_INVOKER() \
    if (g_sgemm_filter.IsMatch("sgemm_sparse")) { \
        for (int i=arg_m.Begin(); i<=arg_m.End(); i<<=1) { \
            for (int j=arg_n.Begin(); j<=arg_n.End(); j<<=1) { \
                for (int k=arg_k.Begin(); k<=arg_k.End(); k<<=1) { \
                    blas_common::sgemm_random_invoker_sparse<float, float>(i, j, k, 1.0f, 1.0f, "sgemm_sparse"); \
                } \
            } \
        } \
    }

#define SGEMM_PRE_INVOKER(prefunc, func) \
    if (g_sgemm_filter.IsMatch(#func)) { \
        for (int i=arg_m.Begin(); i<=arg_m.End(); i<<=1) { \
            for (int j=arg_n.Begin(); j<=arg_n.End(); j<<=1) { \
                for (int k=arg_k.Begin(); k<=arg_k.End(); k<<=1) { \
                    blas_common::sgemm_random_invoker_ex(i, j, k, 1.8f, 1.2f, prefunc, func, #func); \
                } \
            } \
        } \
    }

#define SGEMM_INVOKER_MUTABLE(type_t, result_t, func) \
    if (g_sgemm_filter.IsMatch(TO_STRING(LINK_MULTIPLE(type_t, result_t, func)))) { \
        for (int i=arg_m.Begin(); i<=arg_m.End(); i<<=1) { \
            for (int j=arg_n.Begin(); j<=arg_n.End(); j<<=1) { \
                for (int k=arg_k.Begin(); k<=arg_k.End(); k<<=1) { \
                    blas_common::sgemm_random_invoker<type_t, result_t>(i, j, k, 1.8f, 1.2f, func, TO_STRING(LINK_MULTIPLE(func, type_t, result_t))); \
                } \
            } \
        } \
    }

}
