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

    log_error("%s", "Successfully tested the create_matrix() function! \n");
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
    if(is_equal_matrix(submatrix, expected_submatrix)) {
        log_info("%s", "test_get_sub_matrix passed!\n");
    } else {
        log_error("%s", "test_get_sub_matrix failed!\n");
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
    if(is_equal_matrix(submatrix, expected_submatrix)) {
        log_info("%s", "test_get_sub_matrix_except_column passed!\n");
    } else {
        log_error("%s", "test_get_sub_matrix_except_column failed!\n");
    }
}
void test_serialize_matrix() {
    // Create a Matrix
    Matrix* matrix = create_matrix(2, 2);

    // Initialize the matrix
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->columns; j++) {
            matrix->data[i]->elements[j] = j + 1;
        }
    }

    // Serialize the Matrix to a JSON string
    char *matrixJson = serialize_matrix(matrix);

    // Check if the serialized string is equal to the expected string
    char *expectedJson = "{\"rows\":2,\"columns\":2,\"data\":[[{\"size\":2,\"elements\":[1,2]}],[{\"size\":2,\"elements\":[1,2]}]]}";
    if (strcmp(matrixJson, expectedJson) == 0) {
        printf("test_serializeMatrix passed!\n");
    } else {
        printf("test_serializeMatrix failed!\n");
    }

    // Clean up
    free_matrix(matrix);
    free(matrixJson);
}

void split_matrix_test() {
    Matrix* input_matrix = create_matrix(4, 8); 

    // Initialize the input matrix with some sample values
    for (int i = 0; i < input_matrix->rows; i++) {
        for (int j = 0; j < input_matrix->columns; j++) {
            input_matrix->data[i]->elements[j] = i * input_matrix->columns + j + 1;
        }
    }

    int num_matrices = 2;

    MatrixArray* result = split_matrix(input_matrix, num_matrices);

    assert(result->length == num_matrices);

    for (int i = 0; i < num_matrices; i++) {
        assert(result->array[i]->rows == input_matrix->rows);
        assert(result->array[i]->columns == input_matrix->columns / num_matrices);

        for(int row = 0; row < input_matrix->rows; row ++) {
            for(int col = 0; col < result->array[i]->columns; col++) {
                assert(result->array[i]->data[row]->elements[col] == input_matrix->data[row]->elements[col + (i * result->array[i]->columns)]);
            }
        }
    }

    free_matrix(input_matrix);
    free_matrix_arr(result);

    log_info("split_matrix test passed successfully.");
}
