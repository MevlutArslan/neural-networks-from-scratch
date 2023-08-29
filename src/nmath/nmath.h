#ifndef NMATH_H
#define NMATH_H

#include <math.h>
#include "nvector.h"
#include "nmatrix.h"
#include "../../libraries/logger/log.h"
#include "../helper/constants.h"
#include <assert.h>

typedef enum ElementWiseMatrixOperation {
    ADD, SUBTRACT, MULTIPLY
} ElementWiseMatrixOperation;

// MATRIX MATH OPERATIONS

/*
    @returns NULL if m1.cols does not match m2.rows
    @returns A new Matrix.
*/
Matrix* matrix_product(Matrix* m1, Matrix* m2);

#ifdef __cplusplus
extern "C" {
#endif
    // Multiplication for 2 matrices with different dimensions.
    Matrix* matrix_product_cuda(Matrix* m1, Matrix* m2);
    void matrix_vector_addition_cuda(Matrix* matrix, Vector* vector, Matrix* output);
    

    Matrix* matrix_element_wise_operation_cuda(Matrix* m1, Matrix* m2, ElementWiseMatrixOperation operation);

    /*
        Adds elements of m2 to m1 and stores them in m1
    */
    void matrix_element_wise_operation_inplace_cuda(Matrix* m1, Matrix* m2, ElementWiseMatrixOperation operation);
#ifdef __cplusplus
}
#endif
/*
    @returns NULL if m1's dimensions does not match m2's dimensions
    @returns A new Matrix.
*/
void matrix_addition(Matrix* m1, Matrix* m2, Matrix* output);

/*
    Adds m2 elements to m1 and stores result in m1.
*/

void matrix_addition_inplace(Matrix* m1, Matrix* m2);
/*
    @returns NULL if m1's dimensions does not match m2's dimensions
    @returns A new Matrix.
*/
void matrix_subtraction(Matrix* m1, Matrix* m2, Matrix* output);

/*
    @returns NULL if m1's dimensions does not match m2's dimensions
    @returns A new Matrix.
*/
Matrix* matrix_multiplication(Matrix* m1, Matrix* m2);
void matrix_multiplication_inplace(Matrix* m1, Matrix* m2, Matrix* output);

Matrix* matrix_transpose(Matrix* m);
Matrix* matrix_inverse(Matrix* m);
Matrix* matrix_adjugate(Matrix* m);
Matrix* matrix_cofactor(Matrix* m);
Matrix* matrix_scalar_multiply(Matrix* m, double scalar);

/*
    @returns 0 if m is not square
    @returns A new Matrix.
*/
float matrix_determinant(Matrix* m);

// VECTOR MATH OPERATIONS
Vector* vector_addition(const Vector* v1, const Vector* v2);
void vector_addition_parallel(const Vector* v1, const Vector* v2, Vector* output);

Vector* vector_subtraction(const Vector* v1, const Vector* v2);
double vector_dot_product(const Vector* v1, const Vector* v2);
Vector* vector_multiplication(const Vector* v1, const Vector* v2);

Vector* vector_scalar_multiplication(const Vector* v1, double scalar);
Vector* vector_scalar_subtraction(const Vector* v1, double scalar);

double sum_vector(Vector* vector);


Vector* dot_product(Matrix* matrix, Vector* vector);
void matrix_vector_addition(Matrix* m, Vector* v, Matrix* output);
Matrix* batch_matrix_vector_product(Matrix** matrix_arr, Matrix* matrix, int array_length);

int arg_max_vector(Vector* output);
int arg_max_matrix_row(Matrix* matrix, int row_index);

double column_mean(Matrix* matrix, int columnIndex);
double column_standard_deviation(Matrix* matrix, int columnIndex);

#endif