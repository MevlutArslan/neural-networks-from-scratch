#include "math_tests/matrix_operations.h"
#include "helper_tests/linkedlist_test.h"
#include "matrix_tests/matrix_test.h"
#include "test.h"

void run_tests() {
    testAddToTheEnd();
    testMatrixCreation();
    test_matrix_dot_product();
    test_matrix_addition();
    test_matrix_subtraction();
    test_matrix_multiplication();
    test_matrix_transpose();
    test_matrix_determinant();
    test_matrix_inverse();
    test_matrix_cofactor();
}