#include "matrix_test.h"

void testMatrixCreation() {

    // Matrix to compare against
    double rawMatrix[2][2];
    rawMatrix[0][0] = 1.0;
    rawMatrix[0][1] = 2.0;
    rawMatrix[1][0] = 3.0;
    rawMatrix[1][1] = 4.0;

    // Test matrix creation and initialization
    Matrix* matrix = createMatrix(2, 2);
    
    // Manually set values for testing
    matrix->data[0]->elements[0] = 1.0;
    matrix->data[0]->elements[1] = 2.0;
    matrix->data[1]->elements[0] = 3.0;
    matrix->data[1]->elements[1] = 4.0;
    
    // verify the matrix created by create matrix matches the rawMatrix
    int rows = 2;
    int cols = 2;
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            if (matrix->data[i]->elements[j] != rawMatrix[i][j]) {
                printf("Failed while testing createMatrix() function: \n Mismatch at index [%d][%d]: Expected %.2f, Actual %.2f\n", i, j, rawMatrix[i][j], matrix->data[i]->elements[j]);
            }
        }
    }
    
    // Free the memory allocated for the matrix
    freeMatrix(matrix);

    printf("Successfully tested the createMatrix() function! \n");
}