#include "activation_function_test.h"
#include <stdio.h>

void test_activation_function_relu() {
    int size = 5;
    Vector* input = create_vector(size);
    Vector* expected = create_vector(size);

    // Set input values
    for (int i = 0; i < size; i++) {
        input->elements[i] = i * 2.5;
        expected->elements[i] = (i * 2.5 > 0) ? i * 2.5 : 0;
    }

    // Apply relu activation function to the input vector
    relu(input);
    Vector* output = copy_vector(input);

    // Check if output matches the expected values
    int passed = 1;
    for (int i = 0; i < size; i++) {
        if (output->elements[i] != expected->elements[i]) {
            passed = 0;
            break;
        }
    }

    // Print test result
    if (passed) {
        printf("Relu function test: PASSED\n");
    } else {
        printf("Relu function test: FAILED\n");
    }

    // Clean up memory
    free_vector(input);
    free_vector(expected);
    free_vector(output);
}

void test_activation_function_sigmoid() {
    Vector* input = create_vector(1);
    input->elements[0] = 2.5;
    sigmoid(input);

    double expected = 0.924;
    double output = input->elements[0];

    double threshold = 0.001;  // Tolerance for rounding errors

    if (fabs(output - expected) < threshold) {
        printf("Sigmoid function test: PASSED\n");
    } else {
        printf("Sigmoid function test: FAILED\n");
    }
}

