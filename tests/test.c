#include "math_tests/matrix_operations.h"
#include "matrix_tests/matrix_test.h"
#include "neural_network_tests/activation_function_test.h"
#include "test.h"

void run_tests() {
    testMatrixCreation();
    test_matrix_dot_product();
    test_matrix_addition();
    test_matrix_subtraction();
    test_matrix_multiplication();
    test_matrix_transpose();
    test_matrix_determinant();
    test_matrix_inverse();
    test_matrix_cofactor();
    test_activation_function_relu();
    test_activation_function_sigmoid();
    test_activation_function_softmax();
}