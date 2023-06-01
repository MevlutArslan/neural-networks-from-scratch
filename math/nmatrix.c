#include "nmatrix.h"

Matrix* createMatrix(const int rows, const int cols) {
    // create our matrix by allocating it space on the heap.
    Matrix* matrix = malloc(sizeof(Matrix));
    
    matrix->rows = rows;
    matrix->columns = cols;

    // allocate space for the rows
    matrix->data = malloc(rows * sizeof(double*));

    for(int i = 0; i < matrix->rows; i++) {
        matrix->data[i] = malloc(cols * sizeof(double*));
    }

    return matrix;
}

void printMatrix(const Matrix* matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->columns; j++) {
            printf("%f ", matrix->data[i][j]);
        }
        printf("\n");
    }
}

void freeMatrix(Matrix* matrix){

    for(int i = 0; i < matrix->rows; i++){
        free(matrix->data[i]);
    }

    free(matrix->data);
    free(matrix);
}