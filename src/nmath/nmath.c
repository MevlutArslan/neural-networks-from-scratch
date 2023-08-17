#include "nmath.h"
#include "nmatrix.h"
#include "nvector.h"
#include <string.h>

Matrix* matrix_product(Matrix* m1, Matrix* m2) {
    assert(m1->columns == m2->rows);

    Matrix* output = create_matrix(m1->rows, m2->columns);

    for(int i = 0; i < output->rows; i++){
        for(int j = 0; j < output->columns; j++) {
            for(int k = 0; k < m1->columns; k++) {
                double result = m1->get_element(m1, i, k) * m2->get_element(m2, k, j);
                output->set_element(output, i, j, output->get_element(output, i, j) + result);
            }
        }
    }

    return output;
}

void matrix_product_into(Matrix* m1, Matrix* m2, Matrix* output) {
    // m1.cols has to be equal m2.rows
    assert(output != NULL);
    assert( m1->columns == m2->rows);

    // multiple row by each column
    //  [1, 2]   *  [1, 2, 3] => [1 * 1 + 2 * 4][1 * 2 + 2 * 5][1 * 3 + 2 * 6]
    //  [3, 4]      [4, 5, 6]    [3 * 1 + 4 * 4][3 * 2 + 4 * 5][3 * 2 + 4 * 6]
    for(int i = 0; i < output->rows; i++){
        for(int j = 0; j < output->columns; j++) {
            for(int k = 0; k < m1->columns; k++) {
                double result = m1->get_element(m1, i, k) * m2->get_element(m2, k, j);
                output->set_element(output, i, j, output->get_element(output, i, j) + result);
            }
        }
    }
}


void matrix_addition(Matrix* m1, Matrix* m2, Matrix* output) {
    assert(output != NULL);
    assert(m1->rows == m2->rows && m1->columns == m2->columns);

    for(int i = 0; i < output->rows; i++){
        for(int j = 0; j < output->columns; j++) {
            double result = m1->get_element(m1, i, j) + m2->get_element(m2, i, j);
            output->set_element(output, i, j, result);
        }
    }
}

void matrix_subtraction(Matrix* m1, Matrix* m2, Matrix* output) {
    assert(output != NULL);
    assert(m1->rows == m2->rows && m1->columns == m2->columns);

    for(int i = 0; i < output->rows; i++){
        for(int j = 0; j < output->columns; j++) {
            double result = m1->get_element(m1, i, j) - m2->get_element(m2, i, j);
            output->set_element(output, i, j, result);
        }
    }
}

void matrix_multiplication(Matrix* m1, Matrix* m2, Matrix* output){
    assert(output != NULL);
    assert(m1->rows == m2->rows && m1->columns == m2->columns);

    for(int i = 0; i < output->rows; i++){
        for(int j = 0; j < output->columns; j++) {
            double result = m1->get_element(m1, i, j) * m2->get_element(m2, i, j);
            output->set_element(output, i, j, result);
        }
    }
}

Matrix* matrix_transpose(Matrix* m) {
    // switch dimensions
    Matrix* m3 = create_matrix(m->columns, m->rows);

    for(int i = 0; i < m3->rows; i++){
        for(int j = 0; j < m3->columns; j++) {
            m3->set_element(m3, i, j, m->get_element(m, j, i));
        }
    }

    return m3;
}

Matrix* matrix_inverse(Matrix* m) {
    // M * inverse(M) = I
    // inverse(M) = 1 / det(M) * (adjugate(M))
    // adjugate(M) = C^T
    // Cij = -1^i+j * MMij

    double det = matrix_determinant(m);

    if(det == 0) {
        log_error("%s", "Cannot invert a matrix with determinant 0!");
        return NULL;
    }

    Matrix* adjugateMatrix = matrix_adjugate(m);

    Matrix* inverse = matrix_scalar_multiply(adjugateMatrix, 1.0 / det);

    free_matrix(adjugateMatrix);

    return inverse;
}

Matrix* matrix_adjugate(Matrix* m) {
    // adjugate = C^T
    return matrix_transpose(matrix_cofactor(m));
}


Matrix* matrix_cofactor(Matrix* m){ 
    // Cij = -1^i+j * MMij
    Matrix* c = create_matrix(m->rows, m->columns);

    for(int i = 0; i < c->rows; i++) {
        for(int j = 0; j < c->columns; j++){
            c->set_element(c, i, j, pow(-1, i+j) * matrix_determinant(generate_mini_matrix(m, i, j)));
        }
    }

    return c;
}

Matrix* matrix_scalar_multiply(Matrix* m, double scalar){
    Matrix* result = create_matrix(m->rows, m->columns);

    for(int i = 0; i < result->rows; i++) {
        for(int j = 0; j < result->columns; j++) {
            result->set_element(result, i, j, m->get_element(m, i, j) * scalar);
        }
    }

    return result;
}

float matrix_determinant(Matrix* m) {
    if(is_square(m) == 0) {
        log_error("%s", "Cannot calculate the determinant of a non-square Matrix!");
        return 0;
    }

    // base cases
    if(m->rows == 1 && m->columns == 1) {
        return m->get_element(m, 0, 0);
    }

    if(m->rows == 2 && m->columns == 2) {
        return (m->get_element(m, 0, 0) * m->get_element(m, 1, 1)) - (m->get_element(m, 0, 1) * m->get_element(m, 1, 0));
    }
    // det(M) = Sum(j = 1, n) -1^i+j * Mij * det(matrix excluding ith row and jth col)
    // j => random column
    
    int i = 0;
    int det = 0;
    for (int j = 0; j < m->columns; j++) {
    // Generate mini matrix
        Matrix* miniMatrix = generate_mini_matrix(m, i, j);

        det += pow(-1, i + j) * m->get_element(m, i, j) * matrix_determinant(miniMatrix);

        free_matrix(miniMatrix);
    }

    return det;
}

// VECTOR MATH OPERATIONS

Vector* vector_addition(const Vector* v1, const Vector* v2) {
    if(v1->size != v2->size) {
        log_error("%s", "Size's of the vectors need to match to add two vectors!");
        return NULL;
    }

    Vector* v = create_vector(v1->size);
    v->size = v1->size;
    for(int i = 0; i < v->size; i++) {
        v->elements[i] = v1->elements[i] + v2->elements[i];
    }

    return v;
}

void vector_addition_into(const Vector* v1, const Vector* v2, Vector* output) {
    assert(output != NULL);
    assert(v1->size == v2->size);
    for(int i = 0; i < output->size; i++) {
        output->elements[i] = v1->elements[i] + v2->elements[i];
    }
}

Vector* vector_subtraction(const Vector* v1, const Vector* v2){
    if(v1->size != v2->size) {
        log_error("%s", "Size's of the vectors need to match to subtract two vectors!");
        return NULL;
    }

    Vector* v = create_vector(v1->size);

    for(int i = 0; i < v->size; i++) {
        v->elements[i] = v1->elements[i] - v2->elements[i];
    }

    return v;
}

Vector* vector_multiplication(const Vector* v1, const Vector* v2){
    if(v1->size != v2->size) {
        log_error("%s", "Size's of the vectors need to match to add two vectors!");
        return NULL;
    }

    Vector* v = create_vector(v1->size);

    for(int i = 0; i < v->size; i++) {
        v->elements[i] = v1->elements[i] * v2->elements[i];
    }

    return v;
}

Vector* vector_scalar_multiplication(const Vector* v1, double scalar) {
    Vector* v = create_vector(v1->size);

    for(int i = 0; i < v->size; i++) {
        v->elements[i] = v1->elements[i] * scalar;
    }

    return v;
}

double vector_dot_product(const Vector* v1, const Vector* v2) {
    if(v1->size != v2->size) {
        log_error("%s", "Size's of the vectors need to match to calculate dot product! \n");
        return -1;
    }

    double dot_product = 0;

    for(int i = 0; i < v1->size; i++) {
        dot_product += v1->elements[i] * v2->elements[i];
    }

    return dot_product;
}

double sum_vector(Vector* vector) {
    double sum = 0;

    for(int i = 0; i < vector->size; i++) {
        sum += vector->elements[i];
    }
    
    return sum;
}

Vector* vector_scalar_subtraction(const Vector* v1, double scalar) {
    Vector* v = create_vector(v1->size);

    for(int i = 0; i < v->size; i++) {
        v->elements[i] = v1->elements[i] - scalar;
    }

    return v;
}

Vector* dot_product(Matrix* matrix, Vector* vector) {
    assert(matrix->columns == vector->size);

    Vector* result = create_vector(matrix->rows);
    //each column by each row
    for(int matrixRow = 0; matrixRow < matrix->rows; matrixRow++) {
        double sum = 0.0;
        for(int matrixColumn = 0; matrixColumn < matrix->columns; matrixColumn++) {
            sum += matrix->get_element(matrix, matrixRow, matrixColumn) * vector->elements[matrixColumn];
        }
        result->elements[matrixRow] = sum;
    }

    return result;
}

// void dot_product_into(Matrix)

// @TODO: I need a subrange struct to handle rows efficiently, as get row is too slow.
void matrix_vector_addition(Matrix* m, Vector* v, Matrix* output) {
    assert(output != NULL);
    assert(m->columns == v->size);

    Vector* row_data = create_vector(m->columns);
    Vector* result = create_vector(m->columns);

    for (int i = 0; i < m->rows; i++) {
        // Copy into row_data
        memcpy(row_data->elements, m->data->elements + (i * m->columns), m->columns * sizeof(double));

        // Perform vector addition
        vector_addition_into(row_data, v, result);

        // Copy result into output
        memcpy(output->data->elements + i * m->columns, result->elements, output->columns * sizeof(double));
    }

    // Clean up
    free_vector(row_data);
    free_vector(result);
}

Matrix* batch_matrix_vector_product(Matrix** matrix_arr, Matrix* matrix, int array_length) {
    Matrix* result = create_matrix(matrix->rows, matrix_arr[0]->columns);

    Vector* matrix_row = create_vector(matrix->columns);
    
    for(int i = 0; i < matrix->rows; i++) {
        memcpy(matrix_row->elements, matrix->data->elements + (i * matrix->columns), matrix->columns * sizeof(double));

        Vector* result_row = dot_product(matrix_arr[i], matrix_row);

        memcpy(result->data->elements + (i * result->columns), result_row->elements, result_row->size * sizeof(double));
        free_vector(result_row);
    }

    return result;
}


int arg_max_vector(Vector* output) {
    int maxIndex = 0;
    double max = __DBL_MIN__;
    for(int i = 0; i < output->size; i++) {
        if(output->elements[i] > max) {
            max = output->elements[i];
            maxIndex = i;
        }
    }
    
    return maxIndex;
}

int arg_max_matrix_row(Matrix* matrix, int row_index) {
    // Calculate starting index for the row in the flattened matrix.
    int row_start_index = ROW_START(row_index, matrix->columns);

    // Calculate the ending index for the row, which is starting index plus the total columns.
    int row_end_index = ROW_END(row_start_index, matrix->columns);

    int maxIndex = 0;
    double max = __DBL_MIN__;
    for(int i = row_start_index; i < row_end_index; i++) {
        if(matrix->data->elements[i] > max) {
            max = matrix->data->elements[i];
            maxIndex = i - row_start_index;
        }
    }
    
    return maxIndex;
}

double column_mean(Matrix* matrix, int columnIndex) {
    double sum = 0.0f;

    for(int row = 0; row < matrix->rows; row++) {
        sum += matrix->get_element(matrix, row, columnIndex);
    }

    return sum / matrix->rows;
}

double column_standard_deviation(Matrix* matrix, int columnIndex) {
   double mean = column_mean(matrix, columnIndex);
    double sum_squared_diff = 0.0;

    for(int row = 0; row < matrix->rows; row++) {
        sum_squared_diff += pow(matrix->get_element(matrix, row, columnIndex) - mean, 2);
    }

    return sqrt(sum_squared_diff / matrix->rows);
}