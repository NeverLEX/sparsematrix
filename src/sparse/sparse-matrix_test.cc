#include <iostream>
#include "sparse-matrix.h"

int main(int argc, char *argv[]) {
    sblas::SparseMatrix<uint8, uint8, float> tmp;
    if (!tmp.SelfTest()) {
        std::cout << "SparseMatrix Test fail..." << std::endl;
        return -1;
    } else {
        std::cout << "SparseMatrix Test success..." << std::endl;
        return 0;
    }
    return 0;
}
