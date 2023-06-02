#include "nmath.h"


Matrix* multiply(const Matrix* m1, const Matrix* m2) {
    // m1.rows has to be equal m2.cols
    if( m1->columns != m2->rows) {
        printf("Cannot Multiply M1 and M2. M1's column count \n does not match M2's row count! \n");
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

Matrix* add(const Matrix* m1, const Matrix* m2) {
    if(m1->rows != m2->rows || m1->columns != m2->columns) {
        printf("The sizes of the matrices do not match!");
        return NULL;
    }

    Matrix* m3 = createMatrix(m1->rows, m1->columns);
    for(int i = 0; i < m3->rows; i++){
        for(int j = 0; j < m3->columns; j++) {
            m3->data[i][j] = m1->data[i][j] + m2->data[i][j];
        }
    }

    return m3;
}

Matrix* subtract(const Matrix* m1, const Matrix* m2) {
    if(m1->rows != m2->rows || m1->columns != m2->columns) {
        printf("The sizes of the matrices do not match!");
        return NULL;
    }

    Matrix* m3 = createMatrix(m1->rows, m1->columns);
    for(int i = 0; i < m3->rows; i++){
        for(int j = 0; j < m3->columns; j++) {
            m3->data[i][j] = m1->data[i][j] - m2->data[i][j];
        }
    }

    return m3;
}

Matrix* elementWiseMultiply(const Matrix* m1, const Matrix* m2){
    if(m1->rows != m2->rows || m1->columns != m2->columns) {
        printf("The sizes of the matrices do not match!");
        return NULL;
    }

    Matrix* m3 = createMatrix(m1->rows, m1->columns);
    for(int i = 0; i < m3->rows; i++){
        for(int j = 0; j < m3->columns; j++) {
            m3->data[i][j] = m1->data[i][j] * m2->data[i][j];
        }
    }

    return m3;
}

Matrix* transpose(const Matrix* m) {
    // switch dimensions
    Matrix* m3 = createMatrix(m->columns, m->rows);

    for(int i = 0; i < m3->rows; i++){
        for(int j = 0; j < m3->columns; j++) {
            m3->data[i][j] = m->data[j][i];
        }
    }

    return m3;
}

Matrix* inverse(const Matrix* m) {
    // M * inverse(M) = I
    // inverse(M) = 1 / det(M) * (adjugate(M))
    // adjucate(M) = C^T
    // C = 
    return NULL;
}

float determinant(const Matrix* m) {
    if(isSquare(m) == 0) {
        printf("Cannot calculate the determinant of a non-square Matrix!");
        return 0;
    }

    // base cases
    if(m->rows == 1 && m->columns == 1) {
        return m->data[0][0];
    }

    if(m->rows == 2 && m->columns == 2) {
        return (m->data[0][0] * m->data[1][1]) - (m->data[0][1] * m->data[1][0]);
    }
    // det(M) = Sum(j = 1, n) -1^i+j * Mij * det(matrix excluding ith row and jth col)
    // j => random column
    
    int i = 0;
    int det = 0;
    for(int j = 0; j < m->columns; j++) {
        // create mini matrix
        Matrix* miniMatrix = createMatrix(m->rows - 1, m->columns - 1);
        
        for(int mI = 1; mI < m->rows; mI++) {
            int mJNew = 0;
            for(int mJ = 0; mJ < m->columns; mJ++) {
                if(mJ == j) {
                    continue;
                }
                miniMatrix->data[mI - 1][mJNew] = m->data[mI][mJ];
                mJNew++;
            }
        }

        det += pow(-1, i+j) * m->data[i][j] * determinant(miniMatrix);
        freeMatrix(miniMatrix);
    }

    return det;
}
