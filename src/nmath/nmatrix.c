#include "nmatrix.h"


Matrix* createMatrix(const int rows, const int cols) {
    Matrix* matrix = malloc(sizeof(Matrix));
    
    matrix->rows = rows;
    matrix->columns = cols;

    matrix->data = malloc(rows * sizeof(Vector*));

    for(int i = 0; i < matrix->rows; i++) {
        matrix->data[i] = createVector(cols);
    }

    return matrix;
}

void initializeMatrixWithRandomValuesInRange(Matrix* matrix, double min, double max) {
    for(int row = 0; row < matrix->rows; row++) {
        for(int col = 0; col < matrix->columns; col++) {
            matrix->data[row]->elements[col] = ((double)rand() / (double)RAND_MAX) * (max - min) + min;
        }
    }    
}

void fillMatrix(Matrix* matrix, double value) {
    double* dataPtr = matrix->data[0]->elements;  // Pointer to the first element in the matrix

    size_t dataSize = matrix->rows * matrix->columns * sizeof(double);  // Calculate the size of the data in bytes

    memset(dataPtr, value, dataSize);
}

void freeMatrix(Matrix* matrix){
    for(int i = 0; i < matrix->rows; i++){
        freeVector(matrix->data[i]);
    }

    free(matrix->data);
    free(matrix);
}

void printMatrix(const Matrix* matrix) {
    printf("[");
    for (int i = 0; i < matrix->rows; i++) {
        printf("[ ");
        for (int j = 0; j < matrix->columns; j++) {
            printf("%f ", matrix->data[i]->elements[j]);
        }
        printf("]\n");
    }
    printf("]\n");

}


int isEqual(const Matrix* m1, const Matrix* m2) {
    double epsilon = 1e-6; // Adjust the epsilon value as needed

    if (m1->rows != m2->rows || m1->columns != m2->columns) {
        return 0; // Matrices have different dimensions
    }

    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->columns; j++) {
            double diff = fabs(m1->data[i]->elements[j] - m2->data[i]->elements[j]);
            if (diff > epsilon) {
                return 0; // Element mismatch
            }
        }
    }

    return 1; // Matrices are equal
}


int isSquare(const Matrix* m){
    if(m->rows != m->columns) {
        return 0;
    }

    return 1;
}

Matrix* generateMiniMatrix(const Matrix* m, int excludeRow, int excludeColumn) {
    Matrix* miniMatrix = createMatrix(m->rows - 1, m->columns - 1);
    
    int miniMatrixRow = 0;
    for (int matrixRow = 0; matrixRow < m->rows; matrixRow++) {
        if (matrixRow == excludeRow) {
            continue;
        }
        
        int miniMatrixColumn = 0;
        for (int matrixColumn = 0; matrixColumn < m->columns; matrixColumn++) {
            if (matrixColumn == excludeColumn) {
                continue;
            }
            
            miniMatrix->data[miniMatrixRow]->elements[miniMatrixColumn] = m->data[matrixRow]->elements[matrixColumn];
            miniMatrixColumn++;
        }
        
        miniMatrixRow++;
    }
    
    return miniMatrix;
}