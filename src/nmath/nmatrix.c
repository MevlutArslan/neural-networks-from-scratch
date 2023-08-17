#include "nmatrix.h"
#include "../../libraries/logger/log.h"
#include "nvector.h"
#include <assert.h>
#include <string.h>

Matrix* create_matrix(const int rows, const int cols) {
    Matrix* matrix = malloc(sizeof(Matrix));
    
    matrix->rows = rows;
    matrix->columns = cols;
    
    matrix->set_element = set_element;
    matrix->get_element = get_element;

    matrix->set_row = set_row;
    matrix->get_row = get_row;

    matrix->data = create_vector(rows * cols);

    return matrix;
}

Matrix** create_matrix_arr(int length) {
    Matrix** array = (Matrix**) malloc(length * sizeof(Matrix*));
    for(int i = 0; i < length; i++) {
        array[i] = NULL;
    }
    return array;
}

void set_element(struct Matrix* matrix, int row, int col, double value) {
    matrix->data->elements[FLAT_INDEX(row, col, matrix->columns)] = value;
}

inline double get_element(struct Matrix* matrix, int row, int col) {
    return matrix->data->elements[FLAT_INDEX(row, col, matrix->columns)];
}

void set_row(struct Matrix* matrix, Vector* row, int row_index) {
    if(row_index >= matrix->rows) {
        log_error("Row #%d is out of bounds!", row_index);
    }

    // Calculate starting index for the row in the flattened matrix.
    int row_start_index = ROW_START(row_index, matrix->columns);

    // Calculate the ending index for the row, which is starting index plus the total columns.
    int row_end_index = ROW_END(row_start_index, matrix->columns);

    // index for the passed in vector
    int row_vector_index = 0;

    for(int i = row_start_index; i < row_end_index; i++) {
        matrix->data->elements[i] = row->elements[row_vector_index];
        row_vector_index++;
    }
}

Vector* get_row(struct Matrix* matrix, int row_index) {
    if(row_index >= matrix->rows) {
        log_error("Row #%d is out of bounds!", row_index);
    }
    // Calculate starting index for the row in the flattened matrix.
    int row_start_index = ROW_START(row_index, matrix->columns);

    // Calculate the ending index for the row, which is starting index plus the total columns.
    int row_end_index = ROW_END(row_start_index, matrix->columns);

    // Extract the row from the flattened matrix and returnit.
    return slice_vector(matrix->data, row_start_index, row_end_index);
}

void fill_matrix_random(Matrix* matrix, double min, double max) {
    for(int row = 0; row < matrix->rows; row++) {
        for(int col = 0; col < matrix->columns; col++) {
            matrix->set_element(matrix, row, col, ((double)rand() / (double)RAND_MAX) * (max - min) + min);
        }
    }    
}

void fill_matrix(Matrix* matrix, double value) {
    for(int i = 0; i < matrix->rows; i++) {
        for(int j = 0; j < matrix->columns; j++) {
            matrix->set_element(matrix, i, j, value);
        }
    }
}

void free_matrix(Matrix* matrix){
    if(matrix == NULL) return;
    
    free_vector(matrix->data);
    free(matrix);
}


char* matrix_to_string(const Matrix* matrix) {

    // Initial size for the string
    int size = matrix->rows * matrix->columns * 20;

    char* str = (char*)malloc(size * sizeof(char));

    // Start of the matrix
    strcat(str, "\t\t\t\t");
    strcat(str, "[");
    
    // Initialize the string with a null-terminated character to ensure we start with an empty string
    str[0] = '\0';

    // Loop through each row
    for (int i = 0; i < matrix->rows; i++) {
        if (i > 0) {
            strcat(str, "\n");
            strcat(str, "\t\t\t\t");
        }
        strcat(str, "[");
        // Loop through each column
        for (int j = 0; j < matrix->columns; j++) {

            // Convert the current element to a string
            char element[20];
            sprintf(element, "%f", matrix->get_element(matrix, i, j));

            // Add a comma and space if not at the beginning of the row
            if (j != 0) {
                strcat(str, ", ");
            }
            strcat(str, element);
        }

        strcat(str, "]");
    }
    // End of the matrix
    strcat(str, "]\n");
    return str;
}


int is_equal(const Matrix* m1, const Matrix* m2) {
    double epsilon = 1e-6; // Adjust the epsilon value as needed
    // log_info("m1 dim: (%d, %d), m2 dim: (%d, %d)", m1->rows, m1->columns, m2->rows, m2->columns);
    if (m1->rows != m2->rows || m1->columns != m2->columns) {
        return 0; // Matrices have different dimensions
    }

    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->columns; j++) {
            double diff = fabs(m1->get_element(m1, i, j) - m2->get_element(m2, i, j));
            if (diff > epsilon) {
                return 0; // Element mismatch
            }
        }
    }

    return 1; // Matrices are equal
}


int is_square(const Matrix* m){
    if(m->rows != m->columns) {
        return 0;
    }

    return 1;
}

void shuffle_rows(Matrix* matrix) {
    int numberOfRows = matrix->rows;

    // Step 1: Initialize the permutation array with the original order
    int* permutation = malloc(numberOfRows * sizeof(int));
    for (int i = 0; i < numberOfRows; i++) {
        permutation[i] = i;
    }

    // Step 2: Shuffle the permutation array using the Fisher-Yates algorithm
    for (int i = numberOfRows - 1; i > 0; i--) {
        // Generate a random index
        int j = rand() % (i + 1);

        // Swap permutation[i] and permutation[j]
        int temp = permutation[i];
        permutation[i] = permutation[j];
        permutation[j] = temp;
    }

    // Shuffle the rows directly within the original matrix
    for (int i = 0; i < numberOfRows; i++) {
        if (i != permutation[i]) {
            double* tempRow = malloc(sizeof(double) * matrix->columns);
            memcpy(tempRow, matrix->data->elements + (i * matrix->columns), matrix->columns * sizeof(double));

            // matrix->data[i]->elements = matrix->data[permutation[i]]->elements;
            memcpy( matrix->data->elements + (i * matrix->columns), matrix->data->elements + (permutation[i] * matrix->columns), matrix->columns * sizeof(double));

            // matrix->data[permutation[i]]->elements = tempRow;
            memcpy(matrix->data->elements + (permutation[i] * matrix->columns), tempRow, matrix->columns * sizeof(double));

            free(tempRow);
        }
    }

    // // Clean up
    free(permutation);
}

// @todo fix bugs related to this function (Details on Notion)
Matrix* generate_mini_matrix(const Matrix* m, int excludeRow, int excludeColumn) {
    Matrix* miniMatrix = create_matrix(m->rows - 1, m->columns - 1);
    
    int miniMatrixRow = 0;
    for (int matrixRow = 0; matrixRow < m->rows; matrixRow++) {
        if (matrixRow == excludeRow) {
            continue;
        }
        
        int miniMatrixColumn = 0;
        for (int matrixColumn = 0; matrixColumn < m->columns; matrixColumn++) {
            if (matrixColumn == excludeColumn) {
                continue;
            }
            
            miniMatrix->set_element(miniMatrix, miniMatrixRow, miniMatrixColumn, m->get_element(m, matrixRow, matrixColumn));
            miniMatrixColumn++;
        }
        
        miniMatrixRow++;
    }
    
    return miniMatrix;
}


Matrix* copy_matrix(const Matrix* source) {

  Matrix* matrix = malloc(sizeof(Matrix));
  
  memcpy(matrix, source, sizeof(Matrix));
  
  matrix->data = create_vector(source->rows * source->columns);
  memcpy(matrix->data, source->data, 
         sizeof(double) * source->rows * source->columns);

  return matrix;
}

void copy_matrix_into(const Matrix* source, Matrix* target) {
    assert(source->rows == target->rows && source->columns == target->columns);

    memcpy(target->data->elements, source->data->elements, source->rows * source->columns * sizeof(double));
}

Matrix* get_sub_matrix(Matrix* source, int startRow, int endRow, int startCol, int endCol) {
    int rows = endRow - startRow + 1;
    int cols = endCol - startCol + 1;
    Matrix* matrix = create_matrix(rows, cols);

    for(int i = startRow; i <= endRow; i++) {
        for(int j = startCol; j <= endCol; j++) {
            double value = source->get_element(source, i, j);
            matrix->set_element(matrix, i - startRow, j - startCol, value);
        }
    }

    return matrix;
}

Matrix* get_sub_matrix_except_column(Matrix* source, int startRow, int endRow, int startCol, int endCol, int columnIndex) {
    // if the column we want to remove is out of the range we want our submatrix to be in
    // the column we want to remove can automatically be left out of the submatrix.
    if(columnIndex < startCol || columnIndex > endCol) {
        return get_sub_matrix(source, startRow, endRow, startCol, endCol);
    }

    Matrix* newMatrix = create_matrix(endRow - startRow + 1, (endCol - startCol + 1) - 1);
    for(int row = startRow; row <= endRow; row++) {
        int columnIndexDif = startCol;
        for(int col = startCol; col <= endCol; col++) {
            if(col == columnIndex) {
                columnIndexDif += 1;
                continue;
            }
            newMatrix->set_element(newMatrix, row - startRow, col - columnIndexDif, source->get_element(source, row, col));
        }
    }

    return newMatrix;
}

char* serialize_matrix(const Matrix* matrix) {
    cJSON *root = cJSON_CreateObject();
    cJSON *data = cJSON_CreateArray();

    for (int i = 0; i < matrix->rows; i++) {
        cJSON *row = cJSON_CreateArray();
        
        for(int j = 0; j < matrix->columns; j++) {
            cJSON_AddItemToArray(row, cJSON_CreateNumber(matrix->get_element(matrix, i, j)));
        }
        
        cJSON_AddItemToArray(data, row);
    }

    cJSON_AddItemToObject(root, "rows", cJSON_CreateNumber(matrix->rows));
    cJSON_AddItemToObject(root, "columns", cJSON_CreateNumber(matrix->columns));
    cJSON_AddItemToObject(root, "data", data);

    char *jsonString = cJSON_PrintUnformatted(root);

    cJSON_Delete(root);

    return jsonString;
}

Matrix* deserialize_matrix(cJSON* json) {
    int rows = cJSON_GetObjectItem(json, "rows")->valueint;
    int columns = cJSON_GetObjectItem(json, "columns")->valueint;

    Matrix* matrix = create_matrix(rows, columns);

    cJSON* json_data = cJSON_GetObjectItem(json, "data");

    for (int i = 0; i < rows; i++) {
        cJSON* json_row = cJSON_GetArrayItem(json_data, i);
        
        for(int j = 0; j < columns; j++) {
            double value = cJSON_GetArrayItem(json_row, j)->valuedouble;
            matrix->set_element(matrix, i, j, value);
        }
    }

    return matrix;
}