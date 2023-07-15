#ifndef NMATH_H
#define NMATH_H

#include <math.h>
#include "nvector.h"
#include "nmatrix.h"
#include "../../libraries/logger/log.h"
#include "../helper/constants.h"
// MATRIX MATH OPERATIONS

/*
    @returns NULL if m1.cols does not match m2.rows
    @returns A new Matrix.
*/
Matrix* matrix_product(const Matrix* m1, const Matrix* m2);

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

Matrix* matrix_transpose(const Matrix* m);
Matrix* matrix_inverse(const Matrix* m);
Matrix* matrix_adjucate(const Matrix* m);
Matrix* matrix_cofactor(const Matrix* m);
Matrix* matrix_scalar_multiply(const Matrix* m, const double scalar);

/*
    @returns 0 if m is not square
    @returns A new Matrix.
*/
float matrix_determinant(const Matrix* m);

// VECTOR MATH OPERATIONS
Vector* vector_addition(const Vector* v1, const Vector* v2);
Vector* vector_subtraction(const Vector* v1, const Vector* v2);
double vector_dot_product(const Vector* v1, const Vector* v2);
Vector* vector_multiplication(const Vector* v1, const Vector* v2);

Vector* vector_scalar_multiplication(const Vector* v1, double scalar);
Vector* vector_scalar_subtraction(const Vector* v1, double scalar);

double sum_vector(const Vector* vector);
Vector* dot_product(Matrix* matrix, Vector* vector);

// Conversion
Matrix* vector_to_matrix(const Vector* vector);
Vector* matrix_to_vector(Matrix* matrix);

int arg_max(Vector* output);

double column_mean(Matrix* matrix, int columnIndex);
double column_standard_deviation(Matrix* matrix, int columnIndex);

#endif