#include "matrix_operations.h"

void testMatrixMultiplication() {
    // Create two matrices
    Matrix* m1 = createMatrix(2, 3);
    m1->data[0][0] = 1.0;
    m1->data[0][1] = 2.0;
    m1->data[0][2] = 3.0;
    m1->data[1][0] = 4.0;
    m1->data[1][1] = 5.0;
    m1->data[1][2] = 6.0;

    Matrix* m2 = createMatrix(3, 2);
    m2->data[0][0] = 7.0;
    m2->data[0][1] = 8.0;
    m2->data[1][0] = 9.0;
    m2->data[1][1] = 10.0;
    m2->data[2][0] = 11.0;
    m2->data[2][1] = 12.0;

    // Create the expected result matrix
    Matrix* expected = createMatrix(2, 2);
    expected->data[0][0] = 58.0;
    expected->data[0][1] = 64.0;
    expected->data[1][0] = 139.0;
    expected->data[1][1] = 154.0;

    // Perform matrix multiplication
    Matrix* result = multiply(m1, m2);

    // Compare the result with the expected matrix
    int success = 1;
    for (int i = 0; i < expected->rows; i++) {
        for (int j = 0; j < expected->columns; j++) {
            if (result->data[i][j] != expected->data[i][j]) {
                success = 0;
                break;
            }
        }
    }

    // Print success or failure message
    if (success) {
        printf("Matrix multiplication test succeeded!\n");
    } else {
        printf("Matrix multiplication test failed!\n");
    }

    // Clean up memory
    freeMatrix(m1);
    freeMatrix(m2);
    freeMatrix(expected);
    freeMatrix(result);
}