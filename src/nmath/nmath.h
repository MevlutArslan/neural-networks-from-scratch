#ifndef NMATH_H
#define NMATH_H

#include <math.h>
#include "nvector.h"
#include "nmatrix.h"
#include "../../libraries/logger/log.h"
#include "../helper/constants.h"
#include <assert.h>
#include <pthread.h>

// MATRIX MATH OPERATIONS

/*
    @returns NULL if m1.cols does not match m2.rows
    @returns A new Matrix.
*/
Matrix* matrix_product(const Matrix* m1, const Matrix* m2);
void matrix_product_inplace(const Matrix* m1, const Matrix* m2, Matrix* output);

/*
    @returns NULL if m1's dimensions does not match m2's dimensions
    @returns A new Matrix.
*/
Matrix* matrix_addition(const Matrix* m1, const Matrix* m2);

/*
    @returns NULL if m1's dimensions does not match m2's dimensions
    @returns A new Matrix.
*/
Matrix* matrix_subtraction(const Matrix* m1, const Matrix* m2);

/*
    @returns NULL if m1's dimensions does not match m2's dimensions
    @returns A new Matrix.
*/
Matrix* matrix_multiplication(const Matrix* m1, const Matrix* m2);

Matrix* matrix_scalar_multiply(const Matrix* m, const double scalar);

Matrix* matrix_transpose(const Matrix* m);
Matrix* matrix_inverse(const Matrix* m);
Matrix* matrix_adjucate(const Matrix* m);
Matrix* matrix_cofactor(const Matrix* m);

/*
    @returns 0 if m is not square
    @returns A new Matrix.
*/
float matrix_determinant(const Matrix* m);

// VECTOR MATH OPERATIONS
double vector_product(const Vector* v1, const Vector* v2);

/*
    Writes to the output vector directly instead of creating a new vector in the function.
*/
void vector_addition(const Vector* v1, const Vector* v2, Vector* output);

/*
    Writes to the output vector directly instead of creating a new vector in the function.
*/
void vector_subtraction(const Vector* v1, const Vector* v2, Vector* output);

/*
    Writes to the output vector directly instead of creating a new vector in the function.
*/
void vector_multiplication(const Vector* v1, const Vector* v2, Vector* output);

/*
    Writes to the output vector directly instead of creating a new vector in the function.
*/
void vector_scalar_multiplication(const Vector* v1, double scalar, Vector* output);

/*
    Writes to the output vector directly instead of creating a new vector in the function.
*/
void vector_scalar_subtraction(const Vector* v1, double scalar, Vector* output);

double sum_vector(const Vector* vector);

void dot_product(Matrix* matrix, Vector* vector, Vector* output);
Matrix* matrix_vector_addition(Matrix* matrix, Vector* vector);

Matrix** matrix_product_arr(Matrix** matrix_arr, Matrix* matrix, int size);
Matrix* matrix_vector_product_arr(Matrix** matrix_arr, Matrix* matrix, int size);

// Conversion
Matrix* vector_to_matrix(const Vector* vector);
Vector* matrix_to_vector(Matrix* matrix);

int arg_max(Vector* output);

double column_mean(Matrix* matrix, int columnIndex);
double column_standard_deviation(Matrix* matrix, int columnIndex);


#endif