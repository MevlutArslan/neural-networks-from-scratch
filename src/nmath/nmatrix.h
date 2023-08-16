#ifndef NMATRIX_H
#define NMATRIX_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "nvector.h"
#include "../helper/constants.h"

#define FLAT_INDEX(ROW, COL, WIDTH) (ROW * WIDTH + COL)
#define ROW_START(ROW, WIDTH) (ROW * WIDTH)
#define ROW_END(START_INDEX, WIDTH) (START_INDEX + WIDTH)
typedef struct Matrix{
    int rows;
    int columns;
    Vector* data;

    void (*set_element)(struct Matrix* matrix, int row, int col, double value);
    double (*get_element)(struct Matrix* matrix, int row, int col);

    void (*set_row)(struct Matrix* matrix, Vector* row, int row_index);
    Vector* (*get_row)(struct Matrix* matrix, int row_index);
} Matrix;

Matrix* create_matrix(const int rows, const int cols);

void set_element(struct Matrix* matrix, int row, int col, double value);
double get_element(struct Matrix* matrix, int row, int col);

Vector* get_row(struct Matrix* matrix, int row_index);
void set_row(struct Matrix* matrix, Vector* row, int row_index);

void fill_matrix_random(Matrix* matrix, double min, double max);
void fill_matrix(Matrix* matrix, double value);

void free_matrix(Matrix* matrix);

char* matrix_to_string(const Matrix* matrix);
int is_equal(const Matrix* m1, const Matrix* m2);

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