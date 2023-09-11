#include "activation_function.h"
#include <stdio.h>
#include <math.h>

// in place modification
void relu(Vector* weighted_sums) {
    for (int i = 0; i < weighted_sums->size; i++) {
        weighted_sums->elements[i] = fmax(0, weighted_sums->elements[i]);
    }
}

double relu_derivative(double weighted_sum){
    return weighted_sum > 0 ? 1.0 : 0;
}


void relu_batched(Matrix* weighted_sums) {
    for(int i = 0; i < weighted_sums->rows; i++) {
        relu(weighted_sums->data[i]);
    }
}

Matrix* relu_derivative_batched(Matrix* weighted_sums) {
    Matrix* derivatives = create_matrix(weighted_sums->rows, weighted_sums->columns);

    for(int i = 0; i < weighted_sums->rows; i++) {
        for(int j = 0; j < weighted_sums->columns; j++) {
            derivatives->data[i]->elements[j] = relu_derivative(weighted_sums->data[i]->elements[j]);
        }
    }

    return derivatives;
}

void leaky_relu(Vector* weighted_sums) {
    for (int i = 0; i < weighted_sums->size; i++) {
        if (weighted_sums->elements[i] < 0) {
            weighted_sums->elements[i] *= 0.01;
        }
    }
}

void leaky_relu_batched(Matrix* matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        leaky_relu(matrix->data[i]);
    }
}


double leaky_relu_derivative(double netInput) {
    return netInput > 0.0 ? 1.0 : 0.01;
}


Matrix* leaky_relu_derivative_batched(Matrix* input) {
    Matrix* result = create_matrix(input->rows, input->columns);

    for(int i = 0; i < result->rows; i++) {
        for(int j = 0; j < result->columns; j++) {
            result->data[i]->elements[j] = leaky_relu_derivative(input->data[i]->elements[j]);
        }
    }

    return result;
}

void sigmoid(Vector* inputs) {
    for(int i = 0; i < inputs->size; i++) {
        inputs->elements[i] = 1 / (1 + exp(-1 * inputs->elements[i]));
    }
}

double sigmoid_derivative(double netInput) {
    double sigmoid_x = 1/(1+exp(-1 * netInput));
    return sigmoid_x * (1 - sigmoid_x);
}

/* output[i] = e ^ weightedSums[i]
               --------------------
               sum of all e ^ weightedSums[i -> number of neurons in the output layer]
                                
*/  
void softmax(Vector* inputs) {
    int size = inputs->size;

    double max_value = inputs->elements[0];
    for (int i = 1; i < size; i++) {
        max_value = fmax(max_value, inputs->elements[i]);
    }
    double sum = 0.0;
    Vector* exponentialValues = create_vector(inputs->size);
    for (int i = 0; i < size; i++) {
        exponentialValues->elements[i] = exp(inputs->elements[i] - max_value);
        sum += exponentialValues->elements[i];
    }

    for (int i = 0; i < size; i++) {
        inputs->elements[i] = exponentialValues->elements[i] / sum;
    }

    free_vector(exponentialValues);
}

void softmax_batched(Matrix* matrix) {
    for(int i = 0; i < matrix->rows; i++) {
        softmax(matrix->data[i]);
    }
}

/*
    Derivative of each output of softmax in respect to each weighted sum of output nodes.

    Quotient Rule states d_(x/y) = (d_x * y) - (x * d_y)
                                   ---------------------
                                           y^2
    
    So d_softmaxOutput[i]   (d_softmaxOutput[i] * weightedSum[i]) - (softmaxOutput[i] * d_weightedSum[i])
      ------------------- = -------------------------------------------------------------------------------
       d_weightedSum[i]                            weightedSum[i] ^ 2
*/
Matrix* softmax_derivative(Vector* output) {
    Matrix* jacobian = create_matrix(output->size, output->size);
    // [0.531323, 0.468677]
    for(int i = 0; i < output->size; i++) {
        for(int j = 0; j < output->size; j++) {
            if(i == j) { // 0.531323 * (1 - 0.531323) = 0.24901 (i == 0) | 
                jacobian->data[i]->elements[j] = output->elements[i] * (1 - output->elements[i]);
            }else{ // (-1 * 0.531323) * 0.468677 => -0.249019
                jacobian->data[i]->elements[j] = -1 * output->elements[i] * output->elements[j];
            }
        }
    }

    return jacobian;
}


Matrix** softmax_derivative_batched(Matrix* output) {
    Matrix** jacobian_matrices = create_matrix_arr(output->rows);
    for(int i = 0; i < output->rows; i++) {
        jacobian_matrices[i] = softmax_derivative(output->data[i]);

        #ifdef DEBUG
            log_info("Jacobian matrix #%d: %s", i, matrix_to_string(jacobian_matrices[i]));
        #endif
    }

    return jacobian_matrices;
}

char* get_activation_function_name(const ActivationFunction activation_function) {
    switch (activation_function) {
        case RELU:
            return RELU_STR;
        case LEAKY_RELU:
            return LEAKY_RELU_STR;
        case SOFTMAX:
            return SOFTMAX_STR;
        default:
            return "unrecognized_afn";
    }
}

ActivationFunction get_activation_function_by_name(char* name) {
    if(strcmp(name, RELU_STR) == 0) {
        return RELU;
    }else if(strcmp(name, LEAKY_RELU_STR) == 0) {
        return LEAKY_RELU;
    }else if(strcmp(name, SOFTMAX_STR) == 0) {
        return SOFTMAX;
    }else {
        log_error("Unrecognized activation function name: %s", name);
        return UNRECOGNIZED_AFN;
    }
}