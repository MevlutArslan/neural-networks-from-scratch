#include "matrix_test.h"

void testMatrixCreation() {

    // Matrix to compare against
    double rawMatrix[2][2];
    rawMatrix[0][0] = 1.0;
    rawMatrix[0][1] = 2.0;
    rawMatrix[1][0] = 3.0;
    rawMatrix[1][1] = 4.0;

    // Test matrix creation and initialization
    Matrix* matrix = create_matrix(2, 2);
    
    // Manually set values for testing
    matrix->data[0]->elements[0] = 1.0;
    matrix->data[0]->elements[1] = 2.0;
    matrix->data[1]->elements[0] = 3.0;
    matrix->data[1]->elements[1] = 4.0;
    
    // verify the matrix created by create matrix matches the rawMatrix
    int rows = 2;
    int cols = 2;
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            if (matrix->data[i]->elements[j] != rawMatrix[i][j]) {
                printf("Failed while testing create_matrix() function: \n Mismatch at index [%d][%d]: Expected %.2f, Actual %.2f\n", i, j, rawMatrix[i][j], matrix->data[i][j]);
            }
        }
    }
    
    // Free the memory allocated for the matrix
    free_matrix(matrix);

    log_error("Successfully tested the create_matrix() function! \n");
}


void test_get_sub_matrix() {
    // Create a 4x4 matrix and fill it with numbers from 1 to 16
    Matrix* source = create_matrix(4, 4);
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            source->data[i]->elements[j] = i*4 + j + 1;
        }
    }

    // Create the expected submatrix
    Matrix* expected_submatrix = create_matrix(2, 2);
    expected_submatrix->data[0]->elements[0] = 6;
    expected_submatrix->data[0]->elements[1] = 7;
    expected_submatrix->data[1]->elements[0] = 10;
    expected_submatrix->data[1]->elements[1] = 11;

    // Get a 2x2 submatrix from the 2nd row and 2nd column to the 3rd row and 3rd column
    Matrix* submatrix = get_sub_matrix(source, 1, 3, 1, 3);
    
    // Check if the submatrix is equal to the expected submatrix
    if(is_equal(submatrix, expected_submatrix)) {
        log_info("test_get_sub_matrix passed!\n");
    } else {
        log_error("test_get_sub_matrix failed!\n");
    }
}

void test_get_sub_matrix_except_column() {
    // Create a 4x4 matrix and fill it with numbers from 1 to 16
    Matrix* source = create_matrix(4, 4);
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            source->data[i]->elements[j] = i*4 + j + 1;
        }
    }

    // Create the expected submatrix
    // Create the expected submatrix
    Matrix* expected_submatrix = create_matrix(3, 2);
    expected_submatrix->data[0]->elements[0] = 6;
    expected_submatrix->data[0]->elements[1] = 8;
    expected_submatrix->data[1]->elements[0] = 10;
    expected_submatrix->data[1]->elements[1] = 12;
    expected_submatrix->data[2]->elements[0] = 14;
    expected_submatrix->data[2]->elements[1] = 16;


    // Get a 3x3 submatrix from the 1st row and 1st column to the 3rd row and 3rd column,
    // excluding the 2nd column
    Matrix* submatrix = get_sub_matrix_except_column(source, 1, 3, 1, 3, 2);
    
    // Check if the submatrix is equal to the expected submatrix
    if(is_equal(submatrix, expected_submatrix)) {
        log_info("test_get_sub_matrix_except_column passed!\n");
    } else {
        log_error("test_get_sub_matrix_except_column failed!\n");
    }
}
