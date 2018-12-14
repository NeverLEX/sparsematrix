#include "cblas.h"
#include "blas_test.h"

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
void cblas_sgemm_baseline(int m, int n, int k, float *a, float *b, float *c, float alpha, float beta) {
    const int lda = k, ldb = k, ldc = n;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void cblas_sgemm_precopy(int m, int k, float *a, float *c) {
    const int lda = k;
    cblas_sgemm_precopy(CblasRowMajor, CblasNoTrans, CblasTrans, m, k, a, lda, c);
}

void cblas_sgemm_precopy_baseline(int m, int n, int k, float *a, float *b, float *c, float alpha, float beta) {
    const int lda = k, ldb = k, ldc = n;
    cblas_sgemm_mul(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

int main(int argc, char *argv[]) {
    StepList arg_m("3"), arg_n("1024"), arg_k("1024");
    if (argc >= 2) arg_m.Parse(argv[1]);
    if (argc >= 3) arg_n.Parse(argv[2]);
    if (argc >= 4) arg_k.Parse(argv[3]);
    if (argc >= 5) g_check_matrix = (argv[4][0] == '1');
    if (argc >= 6) g_sgemm_filter.ParseFilter(argv[5]);

    srand(time(nullptr)); // use current time as seed for random generator

    SGEMM_INVOKER(cblas_sgemm_baseline);
    SGEMM_SPARSE_INVOKER();
    SGEMM_PRE_INVOKER(cblas_sgemm_precopy, cblas_sgemm_precopy_baseline);

    g_time_statis.PrintStatis();

    return 0;
}
