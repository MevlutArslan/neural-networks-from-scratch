#include "matrix_operations.h"

void test_matrix_dot_product() {
    // Create two matrices
    Matrix* m1 = createMatrix(2, 3);
    m1->data[0]->elements[0] = 1.0;
    m1->data[0]->elements[1] = 2.0;
    m1->data[0]->elements[2] = 3.0;
    m1->data[1]->elements[0] = 4.0;
    m1->data[1]->elements[1] = 5.0;
    m1->data[1]->elements[2] = 6.0;

    Matrix* m2 = createMatrix(3, 2);
    m2->data[0]->elements[0] = 7.0;
    m2->data[0]->elements[1] = 8.0;
    m2->data[1]->elements[0] = 9.0;
    m2->data[1]->elements[1] = 10.0;
    m2->data[2]->elements[0] = 11.0;
    m2->data[2]->elements[1] = 12.0;

    // Create the expected result matrix
    Matrix* expected = createMatrix(2, 2);
    expected->data[0]->elements[0] = 58.0;
    expected->data[0]->elements[1] = 64.0;
    expected->data[1]->elements[0] = 139.0;
    expected->data[1]->elements[1] = 154.0;

    // Perform matrix multiplication
    Matrix* result = matrix_dot_product(m1, m2);

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

void test_matrix_addition() {
    // Create two matrices with the same dimensions
    Matrix* m1 = createMatrix(2, 2);
    m1->data[0]->elements[0] = 1.0;
    m1->data[0]->elements[1] = 2.0;
    m1->data[1]->elements[0] = 3.0;
    m1->data[1]->elements[1] = 4.0;

    Matrix* m2 = createMatrix(2, 2);
    m2->data[0]->elements[0] = 5.0;
    m2->data[0]->elements[1] = 6.0;
    m2->data[1]->elements[0] = 7.0;
    m2->data[1]->elements[1] = 8.0;

    // Perform matrix addition
    Matrix* result = matrix_addition(m1, m2);

    // Define the expected result
    Matrix* expected = createMatrix(2, 2);
    expected->data[0]->elements[0] = 6.0;
    expected->data[0]->elements[1] = 8.0;
    expected->data[1]->elements[0] = 10.0;
    expected->data[1]->elements[1] = 12.0;

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

void test_matrix_subtraction() {
    // Create two matrices with the same dimensions
    Matrix* m1 = createMatrix(2, 2);
    m1->data[0]->elements[0] = 1.0;
    m1->data[0]->elements[1] = 2.0;
    m1->data[1]->elements[0] = 3.0;
    m1->data[1]->elements[1] = 4.0;

    Matrix* m2 = createMatrix(2, 2);
    m2->data[0]->elements[0] = 5.0;
    m2->data[0]->elements[1] = 6.0;
    m2->data[1]->elements[0] = 7.0;
    m2->data[1]->elements[1] = 8.0;

    // Perform matrix addition
    Matrix* result = matrix_subtraction(m1, m2);

    // Define the expected result
    Matrix* expected = createMatrix(2, 2);
    expected->data[0]->elements[0] = m1->data[0]->elements[0] - m2->data[0]->elements[0];
    expected->data[0]->elements[1] = m1->data[0]->elements[1] - m2->data[0]->elements[1];
    expected->data[1]->elements[0] = m1->data[1]->elements[0] - m2->data[1]->elements[0];
    expected->data[1]->elements[1] = m1->data[1]->elements[1] - m2->data[1]->elements[1];

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

void test_matrix_multiplication() {
    // Create two matrices with the same dimensions
    Matrix* m1 = createMatrix(2, 2);
    m1->data[0]->elements[0] = 1.0;
    m1->data[0]->elements[1] = 2.0;
    m1->data[1]->elements[0] = 3.0;
    m1->data[1]->elements[1] = 4.0;

    Matrix* m2 = createMatrix(2, 2);
    m2->data[0]->elements[0] = 5.0;
    m2->data[0]->elements[1] = 6.0;
    m2->data[1]->elements[0] = 7.0;
    m2->data[1]->elements[1] = 8.0;

    // Perform matrix addition
    Matrix* result = matrix_multiplication(m1, m2);

    // Define the expected result
    Matrix* expected = createMatrix(2, 2);
    expected->data[0]->elements[0] = m1->data[0]->elements[0] * m2->data[0]->elements[0];
    expected->data[0]->elements[1] = m1->data[0]->elements[1] * m2->data[0]->elements[1];
    expected->data[1]->elements[0] = m1->data[1]->elements[0] * m2->data[1]->elements[0];
    expected->data[1]->elements[1] = m1->data[1]->elements[1] * m2->data[1]->elements[1];

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

void test_matrix_transpose() {
    // Create a matrix
    Matrix* m = createMatrix(3, 2);
    m->data[0]->elements[0] = 1.0;
    m->data[0]->elements[1] = 2.0;
    m->data[1]->elements[0] = 3.0;
    m->data[1]->elements[1] = 4.0;
    m->data[2]->elements[0] = 5.0;
    m->data[2]->elements[1] = 6.0;

    // Perform matrix transpose
    Matrix* result = matrix_transpose(m);

    // Define the expected result
    Matrix* expected = createMatrix(2, 3);
    expected->data[0]->elements[0] = 1.0;
    expected->data[0]->elements[1] = 3.0;
    expected->data[0]->elements[2] = 5.0;
    expected->data[1]->elements[0] = 2.0;
    expected->data[1]->elements[1] = 4.0;
    expected->data[1]->elements[2] = 6.0;

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

void test_matrix_determinant() {
    // Create a 3x3 matrix
    Matrix* m = createMatrix(3, 3);
    m->data[0]->elements[0] = 1;
    m->data[0]->elements[1] = 2;
    m->data[0]->elements[2] = 3;
    m->data[1]->elements[0] = 4;
    m->data[1]->elements[1] = 5;
    m->data[1]->elements[2] = 6;
    m->data[2]->elements[0] = 7;
    m->data[2]->elements[1] = 8;
    m->data[2]->elements[2] = 9;

    // Calculate the determinant
    float det = matrix_determinant(m);

    if(det == 0) {
        printf("Matrix determinant test: PASSED\n");
    } else {
        printf("Matrix determinant test: FAILED\n");
    }
    // Free the matrix
    freeMatrix(m);
}

void test_matrix_inverse() {
        // Create a sample matrix
    double data[2][2] = {
        {2, 3},
        {1, 4}
    };

    // Create a Matrix struct and initialize it with the sample data
    Matrix* m = createMatrix(2, 2);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->columns; j++) {
            m->data[i]->elements[j] = data[i][j];
        }
    }

    // Calculate the inverse of the matrix
    Matrix* result = matrix_inverse(m);

    Matrix* expected = createMatrix(m->rows, m->columns);
    expected->data[0]->elements[0] = 0.8;
    expected->data[0]->elements[1] = -0.6;
    expected->data[1]->elements[0] = -0.2;
    expected->data[1]->elements[1] = 0.4;

    if (isEqual(result, expected)) {
        printf("Matrix inverse test: PASSED\n");
    } 
    else {
        printf("Matrix inverse test: FAILED\n");
    }

    // Free the memory allocated for the matrices
    freeMatrix(m);
    freeMatrix(result);
    freeMatrix(expected);
}

void test_matrix_cofactor() {
    // Create a sample matrix
    Matrix* m = createMatrix(3, 3);
    m->data[0]->elements[0] = 1;
    m->data[0]->elements[1] = 2;
    m->data[0]->elements[2] = 3;
    m->data[1]->elements[0] = 4;
    m->data[1]->elements[1] = 5;
    m->data[1]->elements[2] = 6;
    m->data[2]->elements[0] = 7;
    m->data[2]->elements[1] = 8;
    m->data[2]->elements[2] = 2;

    // Compute the cofactor matrix
    Matrix* cofactor = matrix_cofactor(m);

    // Define the expected cofactor matrix
    Matrix* expected = createMatrix(3, 3);
    expected->data[0]->elements[0] = -38;
    expected->data[0]->elements[1] = 34;
    expected->data[0]->elements[2] = -3;
    expected->data[1]->elements[0] = 20;
    expected->data[1]->elements[1] = -19;
    expected->data[1]->elements[2] = 6;
    expected->data[2]->elements[0] = -3;
    expected->data[2]->elements[1] = 6;
    expected->data[2]->elements[2] = -3;

    // Print the result
    if (isEqual(cofactor, expected)) {
        printf("Matrix cofactor test: PASSED\n");
    } else {
        printMatrix(cofactor);
        printf("Matrix cofactor test: FAILED\n");
    }

    // Cleanup - deallocate matrices
    freeMatrix(m);
    freeMatrix(cofactor);
    freeMatrix(expected);
}
