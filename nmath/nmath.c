#include "nmath.h"

Matrix* multiply(Matrix* m1, Matrix* m2) {
    // m1.rows has to be equal m2.cols
    if( m1->columns != m2->rows) {
        printf("CANNOT MULTIPLE M1 and M2. M1's column count \n does not match M2's row count! \n");
        return NULL;
    }
    Matrix* m3 = createMatrix(m1->rows, m2->columns);

    // multiple row by each column
    //  [1, 2]   *  [1, 2, 3] => [1 * 1 + 2 * 4][1 * 2 + 2 * 5][1 * 3 + 2 * 6]
    //  [3, 4]      [4, 5, 6]    [3 * 1 + 4 * 4][3 * 2 + 4 * 5][3 * 2 + 4 * 6]
    for(int i = 0; i < m3->rows; i++){
        for(int j = 0; j < m3->columns; j++) {
            for(int k = 0; k < m1->columns; k++) {
                m3->data[i][j] += m1->data[i][k] * m2->data[k][j];
            }
        }
    }
    return m3;
}
