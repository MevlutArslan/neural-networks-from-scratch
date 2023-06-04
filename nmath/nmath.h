#ifndef NMATH_H
#define NMATH_H

#include "nmatrix.h"
#include <math.h>

Matrix* multiplyMatrices(const Matrix* m1, const Matrix* m2);
Matrix* addMatrices(const Matrix* m1, const Matrix* m2);
Matrix* subtractMatrices(const Matrix* m1, const Matrix* m2);
Matrix* elementWiseMultiply(const Matrix* m1, const Matrix* m2);
Matrix* transpose(const Matrix* m);
Matrix* inverse(const Matrix* m);
Matrix* adjucate(const Matrix* m);
Matrix* coeffienceMatrix(const Matrix* m);

Matrix* multiply_scalar(const Matrix* m, const double scalar);

float determinant(const Matrix* m);

#endif