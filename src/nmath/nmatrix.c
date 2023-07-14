#include "nmatrix.h"
#include "../../libraries/logger/log.h"

Matrix* create_matrix(const int rows, const int cols) {
    Matrix* matrix = malloc(sizeof(Matrix));
    
    matrix->rows = rows;
    matrix->columns = cols;

    matrix->data = malloc(rows * sizeof(Vector*));

    for(int i = 0; i < matrix->rows; i++) {
        matrix->data[i] = malloc(sizeof(double) * cols);
    }

    return matrix;
}

void fill_matrix_random(Matrix* matrix, double min, double max) {
    for(int row = 0; row < matrix->rows; row++) {
        for(int col = 0; col < matrix->columns; col++) {
            matrix->data[row][col] = ((double)rand() / (double)RAND_MAX) * (max - min) + min;
        }
    }    
}

void fill_matrix(Matrix* matrix, double value) {
    double* dataPtr = matrix->data[0];  // Pointer to the first element in the matrix

    size_t dataSize = matrix->rows * matrix->columns * sizeof(double);  // Calculate the size of the data in bytes

    memset(dataPtr, value, dataSize);
}

void free_matrix(Matrix* matrix){
    for(int i = 0; i < matrix->rows; i++){
        free(matrix->data[i]);
    }

    free(matrix->data);
    free(matrix);
}
char* matrixToString(const Matrix* matrix) {
    // Initial size for the string

    int size = matrix->rows * matrix->columns * 20;
    char* str = (char*)malloc(size * sizeof(char));

    // Start of the matrix
    strcat(str, "\t\t\t\t");
    strcat(str, "[");

    // Loop through each row
    for (int i = 0; i < matrix->rows; i++) {
        if (i > 0) {
            strcat(str, "\n");
            strcat(str, "\t\t\t\t");
        }
        strcat(str, "[");
        // Loop through each column
        for (int j = 0; j < matrix->columns; j++) {
            // Convert the current element to a string
            char element[20];
            sprintf(element, "%f", matrix->data[i][j]);

            // Add a comma and space if not at the beginning of the row
            if (j != 0) {
                strcat(str, ", ");
            }
            strcat(str, element);
        }
        strcat(str, "]");
    }
    // End of the matrix
    strcat(str, "]\n");
    return str;
}



int is_equal(const Matrix* m1, const Matrix* m2) {
    double epsilon = 1e-6; // Adjust the epsilon value as needed

    if (m1->rows != m2->rows || m1->columns != m2->columns) {
        return 0; // Matrices have different dimensions
    }

    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->columns; j++) {
            double diff = fabs(m1->data[i][j] - m2->data[i][j]);
            if (diff > epsilon) {
                return 0; // Element mismatch
            }
        }
    }

    return 1; // Matrices are equal
}


int is_square(const Matrix* m){
    if(m->rows != m->columns) {
        return 0;
    }

    return 1;
}

Matrix* generate_mini_matrix(const Matrix* m, int excludeRow, int excludeColumn) {
    Matrix* miniMatrix = create_matrix(m->rows - 1, m->columns - 1);
    
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
            
            miniMatrix->data[miniMatrixRow][miniMatrixColumn] = m->data[matrixRow][matrixColumn];
            miniMatrixColumn++;
        }
        
        miniMatrixRow++;
    }
    
    return miniMatrix;
}