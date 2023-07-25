#include "nmatrix.h"
#include "../../libraries/logger/log.h"

Matrix* create_matrix(const int rows, const int cols) {
    Matrix* matrix = malloc(sizeof(Matrix));
    
    matrix->rows = rows;
    matrix->columns = cols;

    matrix->data = malloc(rows * sizeof(Vector*));

    for (int i = 0; i < matrix->rows; i++) {
        matrix->data[i] = create_vector(cols);
    }

    return matrix;
}

void fill_matrix_random(Matrix* matrix, double min, double max) {
    for(int row = 0; row < matrix->rows; row++) {
        for(int col = 0; col < matrix->columns; col++) {
            matrix->data[row]->elements[col] = ((double)rand() / (double)RAND_MAX) * (max - min) + min;
        }
    }    
}

void fill_matrix(Matrix* matrix, double value) {
    for(int i = 0; i < matrix->rows; i++) {
        for(int j = 0; j < matrix->columns; j++) {
            matrix->data[i]->elements[j] = value;
        }
    }
}

void free_matrix(Matrix* matrix){
    for(int i = 0; i < matrix->rows; i++){
        free_vector(matrix->data[i]);
    }

    free(matrix->data);
    free(matrix);
}


char* matrix_to_string(const Matrix* matrix) {

    // Initial size for the string
    int size = matrix->rows * matrix->columns * 20;

    char* str = (char*)malloc(size * sizeof(char));

    // Start of the matrix
    strcat(str, "\t\t\t\t");
    strcat(str, "[");
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
            sprintf(element, "%f", matrix->data[i]->elements[j]);

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

    if (m1->rows != m2->rows || m1->columns != m2->columns) {
        return 0; // Matrices have different dimensions
    }

    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->columns; j++) {
            double diff = fabs(m1->data[i]->elements[j] - m2->data[i]->elements[j]);
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

    // Create a new matrix to hold the shuffled rows
    Matrix* shuffledMatrix = create_matrix(numberOfRows, matrix->columns);

    // Fill the new matrix with rows in the order specified by the permutation
    for (int i = 0; i < numberOfRows; i++) {
        shuffledMatrix->data[i] = copy_vector(matrix->data[permutation[i]]);
    }

    // Replace the old matrix data with the shuffled data
    for (int i = 0; i < numberOfRows; i++) {
        free_vector(matrix->data[i]);
        matrix->data[i] = shuffledMatrix->data[i];
    }

    // Clean up
    free(shuffledMatrix->data); // Only free the data array, not the vectors it points to
    free(shuffledMatrix);
    free(permutation);
}
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
            
            miniMatrix->data[miniMatrixRow]->elements[miniMatrixColumn] = m->data[matrixRow]->elements[matrixColumn];
            miniMatrixColumn++;
        }
        
        miniMatrixRow++;
    }
    
    return miniMatrix;
}


Matrix* copy_matrix(const Matrix* source) {

  Matrix* matrix = create_matrix(source->rows, source->columns);
  
  memcpy(matrix->data, source->data, 
         sizeof(double) * source->rows * source->columns);

  return matrix;
}

Matrix* get_sub_matrix(Matrix* source, int startRow, int endRow, int startCol, int endCol) {
    Matrix* matrix = create_matrix(endRow - startRow + 1, endCol - startCol + 1); // 
    // lets say my source is of dimensions 8,12 and 
    // I want the part of the matrix from the 2nd row to 10th row and 1st column to 13th col
    for(int i = startRow; i <= endRow; i++) { // i goes 2 -> 10 
        for(int j = startCol; j <= endCol; j++) { // j goes 1 -> 13
            // how do i map i and j to my matrix's index
            matrix->data[i - startRow]->elements[j - startCol] = source->data[i]->elements[j]; 
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
        
            newMatrix->data[row - startRow]->elements[col - columnIndexDif] = source->data[row]->elements[col];
        }
    }

    return newMatrix;
}

char* serialize_matrix(const Matrix* matrix) {
    cJSON *root = cJSON_CreateObject();
    cJSON *data = cJSON_CreateArray();

    for (int i = 0; i < matrix->rows; i++) {
        cJSON *row = cJSON_CreateArray();
        
        char *vectorString = serialize_vector(matrix->data[i]);
        cJSON_AddItemToArray(row, cJSON_CreateRaw(vectorString));
        free(vectorString);
        
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
    Matrix* matrix = malloc(sizeof(Matrix));
    if (matrix == NULL) {
        return NULL;
    }

    matrix->rows = cJSON_GetObjectItem(json, "rows")->valueint;
    matrix->columns = cJSON_GetObjectItem(json, "columns")->valueint;
    matrix->data = malloc(matrix->rows * sizeof(Vector*));

    cJSON* json_data = cJSON_GetObjectItem(json, "data");

    for (int i = 0; i < matrix->rows; i++) {
        cJSON* json_row = cJSON_GetArrayItem(json_data, i);
        cJSON* json_vector = cJSON_GetArrayItem(json_row, 0);
        matrix->data[i] = deserialize_vector(json_vector);
    }

    return matrix;
}
