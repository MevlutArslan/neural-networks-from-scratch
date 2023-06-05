#include "nmath.h"


Matrix* matrix_dot_product(const Matrix* m1, const Matrix* m2) {
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

Matrix* matrix_scalar_multiply(const Matrix* m, const double scalar){
    Matrix* result = createMatrix(m->rows, m->columns);

    for(int i = 0; i < result->rows; i++) {
        for(int j = 0; j < result->columns; j++) {
            result->data[i][j] = m->data[i][j] * scalar;
        }
    }

    return result;
}

Matrix* matrix_addition(const Matrix* m1, const Matrix* m2) {
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

Matrix* matrix_subtraction(const Matrix* m1, const Matrix* m2) {
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

Matrix* matrix_multiplication(const Matrix* m1, const Matrix* m2){
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

Matrix* matrix_transpose(const Matrix* m) {
    // switch dimensions
    Matrix* m3 = createMatrix(m->columns, m->rows);

    for(int i = 0; i < m3->rows; i++){
        for(int j = 0; j < m3->columns; j++) {
            m3->data[i][j] = m->data[j][i];
        }
    }

    return m3;
}

Matrix* matrix_inverse(const Matrix* m) {
    // M * inverse(M) = I
    // inverse(M) = 1 / det(M) * (adjugate(M))
    // adjucate(M) = C^T
    // Cij = -1^i+j * MMij

    double det = matrix_determinant(m);

    if(det == 0) {
        printf("Cannot invert a matrix with determinant 0!");
        return NULL;
    }

    Matrix* adjucateMatrix = matrix_adjugate(m);

    Matrix* inverse = matrix_scalar_multiply(adjucateMatrix, 1.0 / det);

    freeMatrix(adjucateMatrix);

    return inverse;
}

Matrix* matrix_cofactor(const Matrix* m){ 
    // Cij = -1^i+j * MMij
    Matrix* c = createMatrix(m->rows, m->columns);

    for(int i = 0; i < c->rows; i++) {
        for(int j = 0; j < c->columns; j++){
            c->data[i][j] = pow(-1, i+j) * matrix_determinant(generateMiniMatrix(m, i, j));
        }
    }

    return c;
}

Matrix* matrix_adjugate(const Matrix* m) {
    // adjucate = C^T
    return matrix_transpose(matrix_cofactor(m));
}

float matrix_determinant(const Matrix* m) {
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
    for (int j = 0; j < m->columns; j++) {
    // Generate mini matrix
        Matrix* miniMatrix = generateMiniMatrix(m, i, j);

        det += pow(-1, i + j) * m->data[i][j] * matrix_determinant(miniMatrix);

        freeMatrix(miniMatrix);
    }

    return det;
}

// VECTOR MATH OPERATIONS

Vector* vector_addition(const Vector* v1, const Vector* v2) {
    if(v1->size != v2->size) {
        printf("Size's of the vectors need to match to add two vectors!");
        return NULL;
    }

    Vector* v = createVector(v1->size);

    for(int i = 0; i < v->size; i++) {
        v->elements[i] = v1->elements[i] + v2->elements[i];
    }

    return v;
}

Vector* vector_subtraction(const Vector* v1, const Vector* v2){
    if(v1->size != v2->size) {
        printf("Size's of the vectors need to match to add two vectors!");
        return NULL;
    }

    Vector* v = createVector(v1->size);

    for(int i = 0; i < v->size; i++) {
        v->elements[i] = v1->elements[i] - v2->elements[i];
    }

    return v;
}

Vector* vector_multiplication(const Vector* v1, const Vector* v2){
    if(v1->size != v2->size) {
        printf("Size's of the vectors need to match to add two vectors!");
        return NULL;
    }

    Vector* v = createVector(v1->size);

    for(int i = 0; i < v->size; i++) {
        v->elements[i] = v1->elements[i] * v2->elements[i];
    }

    return v;
}

Vector* vector_scalar_multiplication(const Vector* v1, double scalar) {
    Vector* v = createVector(v1->size);

    for(int i = 0; i < v->size; i++) {
        v->elements[i] = v1->elements[i] * scalar;
    }

    return v;
}

double vector_dot_product(const Vector* v1, const Vector* v2) {
    if(v1->size != v2->size) {
        printf("Size's of the vectors need to match to add two vectors!");
        return -1;
    }

    double dot_product = 0;

    for(int i = 0; i < v1->size; i++) {
        dot_product += v1->elements[i] * v2->elements[i];
    }

    return dot_product;
}
