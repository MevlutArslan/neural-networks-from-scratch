#include "loss_function_test.h"


void mean_squared_error_perfect_prediction_test() {
    Matrix* y_values = create_matrix(10, 1);
    fill_matrix(y_values, 10.0f);

    Matrix* predicted_values = create_matrix(10, 1);
    fill_matrix(predicted_values, 10.0f);

    double loss = mean_squared_error(y_values, predicted_values);

    assert(loss == 0.0f);

    free_matrix(y_values);
    free_matrix(predicted_values);
    
    log_info("Mean Squared Error with perfect prediction, test passed successfully.");
}

void mean_squared_error_prediction_with_some_error_test() {
    Matrix* y_values = create_matrix(10, 1);
    fill_matrix(y_values, 10.0f);

    Matrix* predicted_values = create_matrix(10, 1);
    fill_matrix(predicted_values, 8.0f);

    double loss = mean_squared_error(y_values, predicted_values);

    // normally it should be 4.0 but because we divide by 2 after each squaring we check against half of the expected.
    assert(loss == 2.0f); 

    free_matrix(y_values);
    free_matrix(predicted_values);

    log_info("Mean Squared Error with some error, test passed successfully.");
}

void mean_squared_error_derivative_perfect_prediction_test() {
    double target = 10.0;
    double predicted = 10.0;

    double derivative = mean_squared_error_derivative(target, predicted);

    // derivative = predicted - target
    assert(derivative == predicted - target);
    log_info("Mean Squared Error Derivative with perfect prediction, test passed successfully.");
}


void mean_squared_error_derivative_with_some_error_test() {
    double target = 10.0;
    double predicted = 8.0;

    double derivative = mean_squared_error_derivative(target, predicted);

    // derivative = predicted - target i.e. 8 - 10
    assert(derivative == predicted - target);
    log_info("Mean Squared Error Derivative with some error, test passed successfully.");
}
