#include "activation_function_test.h"
#include <stdio.h>

void test_activation_function_relu() {
    double input = 2.5;
    double expected = 2.5;
    double output = relu(input);

    if (output == expected)  {
        printf("Relu function test: PASSED\n");
    } else {
        printf("Relu function test: FAILED\n");
    }
}

void test_activation_function_sigmoid() {
    double input = 2.5;
    double expected = 0.924;
    double output = sigmoid(input);

    double threshold = 0.001;  // Tolerance for rounding errors

    if (fabs(output - expected) < threshold) {
        printf("Sigmoid function test: PASSED\n");
    } else {
        printf("Sigmoid function test: FAILED\n");
    }
}

void test_activation_function_softmax() {
    // Create an input vector with some arbitrary values
    Vector* input = createVector(3);
    input->elements[0] = 1.0;
    input->elements[1] = 2.0;
    input->elements[2] = 3.0;

    // Calculate the expected output values using the softmax formula
    double sum_exp = exp(1.0) + exp(2.0) + exp(3.0);
    double expected[3] = { exp(1.0) / sum_exp, exp(2.0) / sum_exp, exp(3.0) / sum_exp };

    // Apply the softmax function to the input vector
    softmax(input);

    // Check whether the output matches the expected values
    double threshold = 0.001;  // Tolerance for rounding errors
    for (int i = 0; i < input->size; i++) {
        if (fabs(input->elements[i] - expected[i]) >= threshold) {
            printf("Softmax function test: FAILED\n");
            return;
        }
    }

    printf("Softmax function test: PASSED\n");
}