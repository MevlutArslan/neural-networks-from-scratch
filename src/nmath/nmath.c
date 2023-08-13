#include "nmath.h"
#include "math_tasks.h"
#include "nmatrix.h"
#include "nvector.h"
#include <pthread.h>

Matrix* matrix_product(const Matrix* m1, const Matrix* m2) {
    // m1.cols has to be equal m2.rows
    if( m1->columns != m2->rows) {
        log_error("%s", "Cannot Multiply M1 and M2. M1's column count \n does not match M2's row count! \n");
        return NULL;
    }
    Matrix* m3 = create_matrix(m1->rows, m2->columns);

    // multiply each row by all columns
    //  [1, 2]   *  [1, 2, 3] => [1 * 1 + 2 * 4][1 * 2 + 2 * 5][1 * 3 + 2 * 6]
    //  [3, 4]      [4, 5, 6]    [3 * 1 + 4 * 4][3 * 2 + 4 * 5][3 * 2 + 4 * 6]
    for(int i = 0; i < m3->rows; i++){
        for(int j = 0; j < m3->columns; j++) {
            for(int k = 0; k < m1->columns; k++) {
                m3->data[i]->elements[j] += m1->data[i]->elements[k] * m2->data[k]->elements[j];
            }
        }
    }

    return m3;
}


Matrix* matrix_addition(const Matrix* m1, const Matrix* m2) {
    if(m1->rows != m2->rows || m1->columns != m2->columns) {
        log_error("%s", "The sizes of the matrices do not match!");
        return NULL;
    }

    Matrix* m3 = create_matrix(m1->rows, m1->columns);
    for(int i = 0; i < m3->rows; i++){
        for(int j = 0; j < m3->columns; j++) {
            m3->data[i]->elements[j] = m1->data[i]->elements[j] + m2->data[i]->elements[j];
        }
    }

    return m3;
}

Matrix* matrix_subtraction(const Matrix* m1, const Matrix* m2) {
    if(m1->rows != m2->rows || m1->columns != m2->columns) {
        log_error("%s", "The sizes of the matrices do not match!");
        return NULL;
    }

    Matrix* m3 = create_matrix(m1->rows, m1->columns);
    for(int i = 0; i < m3->rows; i++){
        for(int j = 0; j < m3->columns; j++) {
            m3->data[i]->elements[j] = m1->data[i]->elements[j] - m2->data[i]->elements[j];
        }
    }

    return m3;
}

Matrix* matrix_multiplication(const Matrix* m1, const Matrix* m2){
    if(m1->rows != m2->rows || m1->columns != m2->columns) {
        log_error("%s", "The sizes of the matrices do not match!");
        return NULL;
    }

    Matrix* m3 = create_matrix(m1->rows, m1->columns);
    for(int i = 0; i < m3->rows; i++){
        for(int j = 0; j < m3->columns; j++) {
            m3->data[i]->elements[j] = m1->data[i]->elements[j] * m2->data[i]->elements[j];
        }
    }

    return m3;
}

Matrix* matrix_transpose(const Matrix* m) {
    // switch dimensions
    Matrix* m3 = create_matrix(m->columns, m->rows);

    for(int i = 0; i < m3->rows; i++){
        for(int j = 0; j < m3->columns; j++) {
            m3->data[i]->elements[j] = m->data[j]->elements[i];
        }
    }

    return m3;
}

Matrix* matrix_inverse(const Matrix* m) {
    // M * inverse(M) = I
    // inverse(M) = 1 / det(M) * (adjucate(M))
    // adjucate(M) = C^T
    // Cij = -1^i+j * MMij

    double det = matrix_determinant(m);

    if(det == 0) {
        log_error("%s", "Cannot invert a matrix with determinant 0!");
        return NULL;
    }

    Matrix* adjucateMatrix = matrix_adjucate(m);

    Matrix* inverse = matrix_scalar_multiply(adjucateMatrix, 1.0 / det);

    free_matrix(adjucateMatrix);

    return inverse;
}

Matrix* matrix_adjucate(const Matrix* m) {
    // adjucate = C^T
    return matrix_transpose(matrix_cofactor(m));
}


Matrix* matrix_cofactor(const Matrix* m){ 
    // Cij = -1^i+j * MMij
    Matrix* c = create_matrix(m->rows, m->columns);

    for(int i = 0; i < c->rows; i++) {
        for(int j = 0; j < c->columns; j++){
            c->data[i]->elements[j] = pow(-1, i+j) * matrix_determinant(generate_mini_matrix(m, i, j));
        }
    }

    return c;
}

Matrix* matrix_scalar_multiply(const Matrix* m, const double scalar){
    Matrix* result = create_matrix(m->rows, m->columns);

    for(int i = 0; i < result->rows; i++) {
        for(int j = 0; j < result->columns; j++) {
            result->data[i]->elements[j] = m->data[i]->elements[j] * scalar;
        }
    }

    return result;
}

float matrix_determinant(const Matrix* m) {
    if(is_square(m) == 0) {
        log_error("%s", "Cannot calculate the determinant of a non-square Matrix!");
        return 0;
    }

    // base cases
    if(m->rows == 1 && m->columns == 1) {
        return m->data[0]->elements[0];
    }

    if(m->rows == 2 && m->columns == 2) {
        return (m->data[0]->elements[0] * m->data[1]->elements[1]) - (m->data[0]->elements[1] * m->data[1]->elements[0]);
    }
    // det(M) = Sum(j = 1, n) -1^i+j * Mij * det(matrix excluding ith row and jth col)
    // j => random column
    
    int i = 0;
    int det = 0;
    for (int j = 0; j < m->columns; j++) {
    // Generate mini matrix
        Matrix* miniMatrix = generate_mini_matrix(m, i, j);

        det += pow(-1, i + j) * m->data[i]->elements[j] * matrix_determinant(miniMatrix);

        free_matrix(miniMatrix);
    }

    return det;
}

// VECTOR MATH OPERATIONS

void vector_addition(const Vector* v1, const Vector* v2, Vector* output) {
    assert(v1->size == v2->size && v1->size == output->size);

    for(int i = 0; i < output->size; i++) {
        output->elements[i] = v1->elements[i] + v2->elements[i];
    }
}

void vector_subtraction(const Vector* v1, const Vector* v2, Vector* output){
    assert(v1->size == v2->size && v1->size == output->size);

    for(int i = 0; i < output->size; i++) {
        output->elements[i] = v1->elements[i] - v2->elements[i];
    }
}

void vector_multiplication(const Vector* v1, const Vector* v2, Vector* output){
    assert(v1->size == v2->size && v1->size == output->size);

    for(int i = 0; i < output->size; i++) {
        output->elements[i] = v1->elements[i] * v2->elements[i];
    }
}

void vector_scalar_multiplication(const Vector* v1, double scalar, Vector* output) {
    assert(v1->size == output->size);

    for(int i = 0; i < output->size; i++) {
        output->elements[i] = v1->elements[i] * scalar;
    }
}

double vector_product(const Vector* v1, const Vector* v2) {
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

Matrix* vector_to_matrix(const Vector* vector) {
    Matrix* matrix = create_matrix(vector->size, 1);
    
    for (int i = 0; i < vector->size; i++) {
        matrix->data[i]->elements[0] = vector->elements[i];
    }
    
    return matrix;
}


Vector* matrix_to_vector(Matrix* matrix) {
    Vector* vector = create_vector(matrix->rows * matrix->columns);
   
    int vectorIndex = 0;

    for (int row = 0; row < matrix->rows; row++) {
        for (int col = 0; col < matrix->columns; col++) {
            vector->elements[vectorIndex] = matrix->data[row]->elements[col];
            vectorIndex++;
        }
    }

    return vector;
}

double sum_vector(const Vector* vector) {
    double sum = 0;

    for(int i = 0; i < vector->size; i++) {
        sum += vector->elements[i];
    }
    
    return sum;
}

void vector_scalar_subtraction(const Vector* v1, double scalar, Vector* output) {
    assert(v1->size == output->size);

    for(int i = 0; i < output->size; i++) {
        output->elements[i] = v1->elements[i] - scalar;
    }
}

void dot_product(Matrix* matrix, Vector* vector, Vector* output) {
    assert(matrix->columns == vector->size);
    
    //each column by each row
    for(int matrixRow = 0; matrixRow < matrix->rows; matrixRow++) {
        double sum = 0.0;
        for(int matrixColumn = 0; matrixColumn < matrix->columns; matrixColumn++) {
            sum += matrix->data[matrixRow]->elements[matrixColumn] * vector->elements[matrixColumn];
        }
        output->elements[matrixRow] = sum;
    }

}

Matrix* matrix_vector_addition(Matrix* matrix, Vector* vector) {
    Matrix* result_matrix = create_matrix(matrix->rows, matrix->columns);

    for(int i = 0; i < matrix->rows; i++) {
        vector_addition(matrix->data[i], vector, result_matrix->data[i]);
    }

    return result_matrix;
}

int arg_max(Vector* output) {
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

double column_mean(Matrix* matrix, int columnIndex) {
    double sum = 0.0f;

    for(int row = 0; row < matrix->rows; row++) {
        sum += matrix->data[row]->elements[columnIndex];
    }

    return sum / matrix->rows;
}

double column_standard_deviation(Matrix* matrix, int columnIndex) {
   double mean = column_mean(matrix, columnIndex);
    double sum_squared_diff = 0.0;

    for(int row = 0; row < matrix->rows; row++) {
        sum_squared_diff += pow(matrix->data[row]->elements[columnIndex] - mean, 2);
    }

    return sqrt(sum_squared_diff / matrix->rows);
}

Matrix** matrix_product_arr(Matrix** matrix_arr, Matrix* matrix, int size) {
    Matrix** result_arr = create_matrix_arr(size);

    for(int i = 0; i < size; i++) {
        result_arr[i] = matrix_product(matrix_arr[i], matrix);
    }

    return result_arr;
}

Matrix* matrix_vector_product_arr(Matrix** matrix_arr, Matrix* matrix, int size) {
    Matrix* result = create_matrix(matrix->rows, matrix_arr[0]->columns);

    for(int i = 0; i < matrix->rows; i++) {
        dot_product(matrix_arr[i], matrix->data[i], result->data[i]);
    }

    return result;
}

// PARALLELIZED CODE

void parallelized_dot_product(void* args) {
    struct MatrixVectorOperation* data = (struct MatrixVectorOperation*) args;
    assert(data->matrix != NULL && data->vector != NULL && data->output != NULL);
    assert(data->matrix->columns == data->vector->size);

    int beginIndex = data->begin_index;
    int endIndex = data->end_index;

    for(int i = beginIndex; i < endIndex; i++) {
        for(int j = 0; j < data->matrix->columns; j++) {
            data->output->elements[i] += data->matrix->data[i]->elements[j] * data->vector->elements[j];
        }
    }
}

