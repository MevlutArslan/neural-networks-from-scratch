#ifndef NMATH_H
#define NMATH_H

#include "nmatrix.h"

Matrix* multiply(Matrix* m1, Matrix* m2);
Matrix* add(Matrix* m1, Matrix* m2);
Matrix* subtract(Matrix* m1, Matrix* m2);
Matrix* elementWiseMultiply(Matrix* m1, Matrix* m2);

#endif