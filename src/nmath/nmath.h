#ifndef NMATH_H
#define NMATH_H

#include "nmatrix.h"
#include <math.h>
#include "nvector.h"

// MATRIX MATH OPERATIONS
Matrix* matrix_dot_product(const Matrix* m1, const Matrix* m2);
Matrix* matrix_addition(const Matrix* m1, const Matrix* m2);
Matrix* matrix_subtraction(const Matrix* m1, const Matrix* m2);
Matrix* matrix_multiplication(const Matrix* m1, const Matrix* m2);
Matrix* matrix_transpose(const Matrix* m);
Matrix* matrix_inverse(const Matrix* m);
Matrix* matrix_adjugate(const Matrix* m);
Matrix* matrix_cofactor(const Matrix* m);

Matrix* matrix_scalar_multiply(const Matrix* m, const double scalar);

float matrix_determinant(const Matrix* m);


// VECTOR MATH OPERATIONS
Vector* vector_addition(const Vector* v1, const Vector* v2);
Vector* vector_subtraction(const Vector* v1, const Vector* v2);
Vector* vector_scalar_multiplication(const Vector* v1, double scalar);
double vector_dot_product(const Vector* v1, const Vector* v2);
Vector* vector_multiplication(const Vector* v1, const Vector* v2);

#endif