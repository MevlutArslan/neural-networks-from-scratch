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

    // Print success or failure message
    if (isEqual(result, expected)) {
        printf("Matrix multiplication test: PASSED\n");
    } else {
        printf("Matrix multiplication test: FAILED\n");
    }

    // Clean up memory
    freeMatrix(m1);
    freeMatrix(m2);
    freeMatrix(expected);
    freeMatrix(result);
}

void testMatrixAddition() {
    // Create two matrices with the same dimensions
    Matrix* m1 = createMatrix(2, 2);
    m1->data[0][0] = 1.0;
    m1->data[0][1] = 2.0;
    m1->data[1][0] = 3.0;
    m1->data[1][1] = 4.0;

    Matrix* m2 = createMatrix(2, 2);
    m2->data[0][0] = 5.0;
    m2->data[0][1] = 6.0;
    m2->data[1][0] = 7.0;
    m2->data[1][1] = 8.0;

    // Perform matrix addition
    Matrix* result = add(m1, m2);

    // Define the expected result
    Matrix* expected = createMatrix(2, 2);
    expected->data[0][0] = 6.0;
    expected->data[0][1] = 8.0;
    expected->data[1][0] = 10.0;
    expected->data[1][1] = 12.0;

    // Compare the result with the expected matrix
    if (isEqual(result, expected)) {
        printf("Matrix addition test: PASSED\n");
    } else {
        printf("Matrix addition test: FAILED\n");
    }

    // Free the memory allocated for matrices
    freeMatrix(m1);
    freeMatrix(m2);
    freeMatrix(result);
    freeMatrix(expected);
}

void testMatrixSubtraction() {
    // Create two matrices with the same dimensions
    Matrix* m1 = createMatrix(2, 2);
    m1->data[0][0] = 1.0;
    m1->data[0][1] = 2.0;
    m1->data[1][0] = 3.0;
    m1->data[1][1] = 4.0;

    Matrix* m2 = createMatrix(2, 2);
    m2->data[0][0] = 5.0;
    m2->data[0][1] = 6.0;
    m2->data[1][0] = 7.0;
    m2->data[1][1] = 8.0;

    // Perform matrix addition
    Matrix* result = subtract(m1, m2);

    // Define the expected result
    Matrix* expected = createMatrix(2, 2);
    expected->data[0][0] = m1->data[0][0] - m2->data[0][0];
    expected->data[0][1] = m1->data[0][1] - m2->data[0][1];
    expected->data[1][0] = m1->data[1][0] - m2->data[1][0];
    expected->data[1][1] = m1->data[1][1] - m2->data[1][1];

    // Compare the result with the expected matrix
    if (isEqual(result, expected)) {
        printf("Matrix subtraction test: PASSED\n");
    } else {
        printf("Matrix subtraction test: FAILED\n");
    }

    // Free the memory allocated for matrices
    freeMatrix(m1);
    freeMatrix(m2);
    freeMatrix(result);
    freeMatrix(expected);
}

void testMatrixElementWiseMultiplication() {
    // Create two matrices with the same dimensions
    Matrix* m1 = createMatrix(2, 2);
    m1->data[0][0] = 1.0;
    m1->data[0][1] = 2.0;
    m1->data[1][0] = 3.0;
    m1->data[1][1] = 4.0;

    Matrix* m2 = createMatrix(2, 2);
    m2->data[0][0] = 5.0;
    m2->data[0][1] = 6.0;
    m2->data[1][0] = 7.0;
    m2->data[1][1] = 8.0;

    // Perform matrix addition
    Matrix* result = elementWiseMultiply(m1, m2);

    // Define the expected result
    Matrix* expected = createMatrix(2, 2);
    expected->data[0][0] = m1->data[0][0] * m2->data[0][0];
    expected->data[0][1] = m1->data[0][1] * m2->data[0][1];
    expected->data[1][0] = m1->data[1][0] * m2->data[1][0];
    expected->data[1][1] = m1->data[1][1] * m2->data[1][1];

    // Compare the result with the expected matrix

    if (isEqual(result, expected)) {
        printf("Matrix element wise multiplication test: PASSED\n");
    } else {
        printf("Matrix element wise multiplication test: FAILED\n");
    }

    // Free the memory allocated for matrices
    freeMatrix(m1);
    freeMatrix(m2);
    freeMatrix(result);
    freeMatrix(expected);
}

void testMatrixTranspose() {
    // Create a matrix
    Matrix* m = createMatrix(3, 2);
    m->data[0][0] = 1.0;
    m->data[0][1] = 2.0;
    m->data[1][0] = 3.0;
    m->data[1][1] = 4.0;
    m->data[2][0] = 5.0;
    m->data[2][1] = 6.0;

    // Perform matrix transpose
    Matrix* result = transpose(m);

    // Define the expected result
    Matrix* expected = createMatrix(2, 3);
    expected->data[0][0] = 1.0;
    expected->data[0][1] = 3.0;
    expected->data[0][2] = 5.0;
    expected->data[1][0] = 2.0;
    expected->data[1][1] = 4.0;
    expected->data[1][2] = 6.0;

    // Compare the result with the expected matrix
    if (isEqual(result, expected)) {
        printf("Matrix transpose test: PASSED\n");
    } else {
        printf("Matrix transpose test: FAILED\n");
    }

    // Free the memory allocated for matrices
    freeMatrix(m);
    freeMatrix(result);
    freeMatrix(expected);
}