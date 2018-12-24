#include <iostream>
#include <cmath>
#include "kernel.h"

template<typename type_t>
void naive_trans(type_t* a, int m, int n, int lda, type_t* sa, int ldsa) {
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            sa[j*ldsa + i] = a[i*lda + j];
        }
    }
}

template<typename type_t>
void gen_matrix_random(type_t *&a, int m, int n) {
    a = (type_t*) malloc (m*n*sizeof(type_t));
    for (int i=0; i<m*n; i++) {
        a[i] = type_t((rand() - RAND_MAX/2));
        if (a[i] > 1000 || a[i] < -1000) a[i] = a[i]/500;
        if (a[i] > 1000 || a[i] < -1000) a[i] = a[i]/500;
        if (a[i] > 1000 || a[i] < -1000) a[i] = a[i]/200;
        if (a[i] > 1000 || a[i] < -1000) a[i] = a[i]/100;
        if (a[i] > 1000 || a[i] < -1000) a[i] = a[i]/100;
    }
}

int main(int argc, char *argv[]) {
    float *a = nullptr, *b = nullptr, *c = nullptr;
    gen_matrix_random(a, 1024, 512);
    gen_matrix_random(b, 1024, 512);
    gen_matrix_random(c, 1024, 512);

    const int m = 1023, n = 511, lda = 512, ldsa = 1024;
    naive_trans(a, m, n, lda, b, ldsa);
    sblas_trans_kernel(a, m, n, lda, c, ldsa);
    bool success = true;
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            if (c[i*ldsa + j] != b[i*ldsa + j]) {
                success = false;
                break;
            }
        }
        if (!success) break;
    }
    int ret = 0;
    if (!success) {
        std::cout << "Kernel Test fail..." << std::endl;
        ret = -1;
    } else {
        std::cout << "Kernel Test success..." << std::endl;
    }
    if (a) free(a), a = nullptr;
    if (b) free(b), b = nullptr;
    if (c) free(c), c = nullptr;
    return ret;
}

