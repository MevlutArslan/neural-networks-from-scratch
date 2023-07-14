#ifndef NMATRIX_H
#define NMATRIX_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "nvector.h"

typedef struct {
    int rows;
    int columns;
    double** data;
} Matrix;

Matrix* create_matrix(const int rows, const int cols);

void fill_matrix_random(Matrix* matrix, double min, double max);
void fill_matrix(Matrix* matrix, double value);

void free_matrix(Matrix* matrix);

char* matrix_to_string(const Matrix* matrix);

int is_equal(const Matrix* m1, const Matrix* m2);

int is_square(const Matrix* m);

// not sure what to call this.
// in the formulas for calculating determinants there is the step of working 
// on a part of the matrix which is obtatined by excluding current row and current column
Matrix* generate_mini_matrix(const Matrix* m, int excludeRow, int excludeColumn);

#endif
