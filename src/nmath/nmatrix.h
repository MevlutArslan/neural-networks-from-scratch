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
    Vector** data;
} Matrix;

Matrix* create_matrix(const int rows, const int cols);
Matrix** create_matrix_arr(int size);
void fill_matrix_random(Matrix* matrix, double min, double max);
void fill_matrix(Matrix* matrix, double value);

void free_matrix(Matrix* matrix);

char* matrix_to_string(const Matrix* matrix);
int is_equal_matrix(const Matrix* m1, const Matrix* m2);

int is_square(const Matrix* m);

void shuffle_rows(Matrix* matrix);

Matrix* copy_matrix(const Matrix* source);

// not sure what to call this.
// in the formulas for calculating determinants there is the step of working 
// on a part of the matrix which is obtatined by excluding current row and current column
Matrix* generate_mini_matrix(const Matrix* m, int excludeRow, int excludeColumn);

/*
    Gets submatrix with in the specified boundaries.
*/
Matrix* get_sub_matrix(Matrix* source, int startRow, int endRow, int startCol, int endCol);

/*
    Gets submatrix with in the specified boundaries and skips a column.
    Both endRow & endCol are inclusive
*/
Matrix* get_sub_matrix_except_column(Matrix* source, int startRow, int endRow, int startCol, int endCol, int columnIndex);

char* serialize_matrix(const Matrix* matrix);
Matrix* deserialize_matrix(cJSON* json);

#endif
