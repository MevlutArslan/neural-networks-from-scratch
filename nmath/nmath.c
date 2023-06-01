#include "nmath.h"

Matrix* multiply(Matrix* m1, Matrix* m2) {
    // m1.rows has to be equal m2.cols
    if( m1->columns != m2->rows) {
        printf("CANNOT MULTIPLE M1 and M2. M1's column count \n does not match M2's row count! \n");
        return NULL;
    }

    Matrix* m3 = malloc(sizeof(Matrix));
    
    // allocate space for rows
    m3->data = malloc(m1->rows * sizeof(double*));
    for(int i = 0; i < m1->rows; i++) {
        m3->data[i] = malloc(m2->columns * sizeof(double*)); 
    }
    // allocate space for cols;


    return m3;
}
