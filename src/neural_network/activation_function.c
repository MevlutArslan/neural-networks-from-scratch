#include "activation_function.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/types.h>



// in place modification
void relu(Vector* vector) {
    for (int i = 0; i < vector->size; i++) {
        vector->elements[i] = fmax(0, vector->elements[i]);
    }
}

double relu_derivative(double netInput){
    return netInput > 0 ? 1.0 : 0;
}

void leakyReluMatrix(Matrix* matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        // leakyRelu(matrix->data[i]);
    }
}

void leakyRelu(Vector* vector) {
    for (int i = 0; i < vector->size; i++) {
        if (vector->elements[i] < 0) {
            vector->elements[i] *= 0.01;
        }
    }
}

double leakyRelu_derivative(double netInput) {
    return netInput > 0.0 ? 1.0 : 0.01;
}


Matrix* leakyRelu_derivative_matrix(Matrix* input) {
    Matrix* result = create_matrix(input->rows, input->columns);

    for(int i = 0; i < result->rows; i++) {
        for(int j = 0; j < result->columns; j++) {
            result->set_element(result, i, j, leakyRelu_derivative(input->get_element(input, i, j)));
            // result->data[i]->elements[j] = leakyRelu_derivative(input->get_element(input, i, j));
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

void softmax_matrix(Matrix* matrix) {
    Vector* matrix_row = create_vector(matrix->columns);
    for(int i = 0; i < matrix->rows; i++) {
        memcpy(matrix_row->elements,  matrix->data->elements + (i * matrix->columns), matrix->columns * sizeof(double));
        softmax(matrix_row);
        memcpy(matrix->data->elements + (i * matrix->columns), matrix_row->elements, matrix->columns * sizeof(double));
    }

    free_vector(matrix_row);
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
                jacobian->set_element(jacobian, i, j, output->elements[i] * (1 - output->elements[i]));
                // jacobian->data[i]->elements[j] = output->elements[i] * (1 - output->elements[i]);
            }else{ // (-1 * 0.531323) * 0.468677 => -0.249019
                jacobian->set_element(jacobian, i, j, -1 * output->elements[i] * output->elements[j]);
                // jacobian->data[i]->elements[j] = -1 * output->elements[i] * output->elements[j];
            }
        }
    }

    return jacobian;
}

void softmax_derivative_parallelized(void* args) {
    SoftmaxDerivativeArgs* softmax_args = (SoftmaxDerivativeArgs*) args;
    Vector* output = softmax_args->output_row;
    Matrix* jacobian = softmax_args->jacobian_matrix;

    for(int i = 0; i < output->size; i++) {
        for(int j = 0; j < output->size; j++) {
            if(i == j) { // 0.531323 * (1 - 0.531323) = 0.24901 (i == 0) | 
                jacobian->set_element(jacobian, i, j, output->elements[i] * (1 - output->elements[i]));
                // jacobian->data[i]->elements[j] = output->elements[i] * (1 - output->elements[i]);
            }else{ // (-1 * 0.531323) * 0.468677 => -0.249019
                jacobian->set_element(jacobian, i, j, -1 * output->elements[i] * output->elements[j]);
                // jacobian->data[i]->elements[j] = -1 * output->elements[i] * output->elements[j];
            }
        }
    }
}

Matrix** softmax_derivative_batched(Matrix* output, struct ThreadPool* thread_pool) {
    Matrix** jacobian_matrices = create_matrix_arr(output->rows);

    Vector** matrix_rows = create_vector_arr(output->rows);
    struct Task** tasks = (struct Task**) calloc(output->rows, sizeof(struct Task*));
    SoftmaxDerivativeArgs args[output->rows];

    for(int i = 0; i < output->rows; i++) {
        jacobian_matrices[i] = create_matrix(output->columns, output->columns);
        matrix_rows[i] = create_vector(output->columns);
        memcpy(matrix_rows[i]->elements, output->data->elements + (i * output->columns), output->columns * sizeof(double));
        
        args[i].jacobian_matrix = jacobian_matrices[i];
        args[i].output_row = matrix_rows[i];

        tasks[i] = create_task(softmax_derivative_parallelized, &args[i]);
        thread_pool->push_task(thread_pool, tasks[i]);
        #ifdef DEBUG
            printf("%s \n", matrix_to_string(jacobian_matrices[i]));
        #endif
    }
    wait_for_all_tasks(thread_pool);

    for(int i = 0; i < output->rows; i++) {
        free(tasks[i]);
        free(matrix_rows[i]);
    }
    
    free(matrix_rows);
    free(tasks);

    return jacobian_matrices;
}

const char* get_activation_function_name(const ActivationFunction activation_function) {
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