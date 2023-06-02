#ifndef NMATH_H
#define NMATH_H

#include "nmatrix.h"
#include <math.h>

Matrix* multiply(const Matrix* m1, const Matrix* m2);
Matrix* add(const Matrix* m1, const Matrix* m2);
Matrix* subtract(const Matrix* m1, const Matrix* m2);
Matrix* elementWiseMultiply(const Matrix* m1, const Matrix* m2);
Matrix* transpose(const Matrix* m);
Matrix* inverse(const Matrix* m);
float determinant(const Matrix* m);

#endif