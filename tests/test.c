#include "math_tests/matrix_operations.h"
#include "matrix_tests/matrix_test.h"
#include "neural_network_tests/activation_function_test.h"
#include "neural_network_tests/optimization_algorithms_test.h"
#include "neural_network_tests/loss_function_test.h"
#include "preprocessing_tests/transformer_preprocessing_test.h"
#include "test.h"

void run_tests() {
    srand(306);
    // testMatrixCreation();
    // test_get_sub_matrix();
    // test_get_sub_matrix_except_column();
    // test_matrix_product();
    // test_matrix_addition();
    // test_matrix_subtraction();
    // test_matrix_multiplication();
    // test_matrix_transpose();
    // test_matrix_determinant();
    // test_matrix_inverse();
    // test_matrix_cofactor();
    // test_activation_function_relu();
    // test_activation_function_sigmoid();
    // test_serialize_vector();
    // test_serialize_matrix();

    test_mock_sgd();
    test_mock_adagrad();
    test_mock_rms_prop();
    test_mock_adam();

    mean_squared_error_perfect_prediction_test();
    mean_squared_error_prediction_with_some_error_test();
    mean_squared_error_derivative_perfect_prediction_test();
    mean_squared_error_derivative_with_some_error_test();

    line_to_embedding_test();
    word_to_embedding_test();
    empty_word_to_embedding_test();
    fill_tokenizer_vocabulary_test();
}