#include "data_processing.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

// length of mnist rows
#define MAX_LINE_LENGTH 4500
#define MAX_TOKENS 256
#define DELIMITER ","

Data* load_csv(char* file_location) {
    FILE* file = fopen(file_location, "r");
    if (file == NULL) {
        printf("Failed to open file: %s\n", file_location);
        return NULL;
    }

    Data* data = malloc(sizeof(Data));
    if (data == NULL) {
        printf("Failed to allocate memory for Data\n");
        fclose(file);
        return NULL;
    }

    data->rows = getRowCount(file_location) - 1;
    data->columns = getColumnCount(file_location);

    data->data = create_matrix(data->rows, data->columns);
    if (data->data == NULL) {
        printf("Failed to create matrix\n");
        fclose(file);
        free(data);
        return NULL;
    }

    char currentLine[MAX_LINE_LENGTH];
    int rowIndex = 0;

    // while there are lines to read
    while (fgets(currentLine, sizeof(currentLine), file) != NULL  && rowIndex < data->rows + 1) {

        #ifdef DEBUG
        if (feof(file)) {
            printf("End of file reached at row index: %d\n", rowIndex);
            break;
        }
        #endif

        if(currentLine[0] == '\n' || (currentLine[0] == '\r' && currentLine[1] == '\n')) {
            continue;
        }

        char* token;
        char* rest = currentLine;
        int colIndex = 0;

        // strtok separates a line by the delimiter and writes the remaining of the line into the address provided
        // we look a step ahead in the while loop but we assign that to the correct index of the matrix
        while ((token = strtok_r(rest, DELIMITER, &rest)) && colIndex < data->columns) {
            if (rowIndex == 0) {
                break;
            }
            double value = atof(token);

            data->data->data[rowIndex - 1]->elements[colIndex] = value;

            // log_debug("at column index: %d", colIndex);
            colIndex++;
        }

        rowIndex++;
    }

    fclose(file);
   
    return data;
}

Matrix** load_text_as_embedding(char* file_location, map_t char_int_map, map_t int_char_map, int max_sequence_length, int vocab_size) {
    FILE* file = fopen(file_location, "r");
    assert(file != NULL);

    char line[3000]; // change this value as you see fit

    int num_rows = getRowCount(file_location);

    Matrix** embeddings = create_matrix_arr(num_rows);
    int* index = malloc(1 * sizeof(int));
    
    int sentence_index = 0;
    while (fgets(line, sizeof(line), file)) {  
        //check for empty lines or end of file
        if (line[0] == '\0' || line[0] == '\n' || feof(file)) {
            // Handle empty lines or end of file, e.g., by skipping them
            continue;
        }
        // log_info("text:%s", line);
        fill_tokenizer_vocabulary(&line, char_int_map, int_char_map, index);
        embeddings[sentence_index] = line_to_embedding(&line, max_sequence_length, vocab_size, char_int_map);
        sentence_index++;
    }
    print_hashmap(char_int_map, CHAR_INT);
    // print_hashmap(int_char_map, INT_CHAR);

    free(index);

    fclose(file);

    return embeddings;
}

Matrix* line_to_embedding(char* text, int max_sequence_length, int vocab_size, map_t char_int_map) {
    Matrix* sentence_embedding = create_matrix(max_sequence_length, vocab_size);
    
    char* token = strtok(text, " ");
    int row_index = 0;

    while (token != NULL && row_index < sentence_embedding->rows) {
        if(row_index == 0 || row_index % 2 == 0) {
            word_to_embedding(token, char_int_map, sentence_embedding->data[row_index]);
            token = strtok(NULL, " "); // Move to the next token
        }else {
            word_to_embedding(" ", char_int_map, sentence_embedding->data[row_index]);
        }
        
        row_index++;
    }

    log_info("row index: %d", row_index);

    return sentence_embedding;
}

void word_to_embedding(char* token, map_t char_int_map, Vector* vector) {
    any_t value;

    if(strcmp(token, " ") == 0) {
        int rc = hashmap_get(char_int_map, token, &value);
        assert(rc == MAP_OK);

        vector->elements[0] = (double) (int) value;
        return;
    }
    
    char* stringified_token = malloc(2 * sizeof(char));
    for(int i = 0; i < strlen(token); i++) {
        if (token[i] == '\n' || token[i] == '\r') {
            break;
        }

        sprintf(stringified_token, "%c", token[i]);


        int rc = hashmap_get(char_int_map, stringified_token, &value);
        assert(rc == MAP_OK);

        vector->elements[i] = (double) (int) value;
    }
    free(stringified_token);
    // log_info("embedding for word: %s is vector: %s", token, vector_to_string(vector));
}

void fill_tokenizer_vocabulary(char* text, map_t char_int_map, map_t int_char_map, int* index) {
    for (int i = 0; i < strlen(text); i++) {
        if (text[i] == '\n' || (text[i] == '\r' && text[i] == '\n')) {
            break;
        }

        char* key = (char*)malloc(2 * sizeof(char)); // Allocate space for the character and null terminator
        sprintf(key, "%c", text[i]); // Initialize the key with the character
       
        any_t return_value;

        int rc = hashmap_get(char_int_map, key, &return_value);
        if (rc == MAP_OK) {
            continue;
        }

        hashmap_put(char_int_map, key, index[0]);

        // Convert index to a string for int_char_map
        char* index_str = (char*)malloc(8 * sizeof(char)); // Assuming a maximum of 8 digits for the index
        assert(index_str != NULL);

        sprintf(index_str, "%d", index[0]);

        char* val = malloc(2 * sizeof(char));
        sprintf(val, "%c", text[i]);

        hashmap_put(int_char_map, index_str, val);

        index[0]++;
    }
}


Vector* extractYValues(Matrix* matrix, int column_index) {
    Vector* yValues = create_vector(matrix->rows);

    for(int i = 0; i < yValues->size; i++) {
        yValues->elements[i] = matrix->data[i]->elements[column_index];
        matrix->data[i]->elements[column_index] = 0.0f;
    }

    return yValues;
}

int getColumnCount(char* file_location) {
    FILE* file = fopen(file_location, "r");
    if(file == NULL) {
        printf("Failed to open file: %s\n", file_location);
        return 0;
    }

    int columnCount = 0;
    char currentLine[MAX_LINE_LENGTH];
    if (fgets(currentLine, sizeof(currentLine), file) != NULL) {
        // Tokenize the first line to count the number of columns
        char* token;
        char* rest = currentLine;
        while ((token = strtok_r(rest, DELIMITER, &rest))) {
            columnCount++;
        }
    }

    fclose(file);
    return columnCount;
}

int getRowCount(char* file_location) {
    FILE* file = fopen(file_location, "r");
    if(file == NULL) {
        printf("Failed to open file: %s\n", file_location);
        return 0;
    }

    int rowCount = 0;
    char currentLine[MAX_LINE_LENGTH];
    while (fgets(currentLine, sizeof(currentLine), file) != NULL) {
        if(currentLine[0] == '\n' || (currentLine[0] == '\r' && currentLine[1] == '\n')) {
            continue;
        }

        rowCount++;
    }

    fclose(file);
    return rowCount;
}


void normalize_column(NormalizationMethod normalization_method, Matrix* matrix, int column_index, float division_factor) {
    switch (normalization_method)
    {
    case STANDARD_DEVIATION:
        normalize_column_standard_deviation(matrix, column_index);
        break;
    case BY_DIVISION:
        normalize_column_by_division(matrix, column_index, division_factor);
    default:
        break;
    }
}

void normalize_column_standard_deviation(Matrix* matrix, int column_index) {
    double mean = column_mean(matrix, column_index);
    double standard_deviation = column_standard_deviation(matrix, column_index);

    for(int row = 0; row < matrix->rows; row++) {
        matrix->data[row]->elements[column_index] = (matrix->data[row]->elements[column_index] - mean) / standard_deviation;
    }
}

void normalize_column_by_division(Matrix* matrix, int column_index, float division_factor) {
    for(int row = 0; row < matrix->rows; row++) {
        matrix->data[row]->elements[column_index] = matrix->data[row]->elements[column_index] / division_factor;
    }
}


void normalizeVector(Vector* vector) {
    double maxValueOfVector = DBL_MIN;
    double minValueOfVector = DBL_MAX;
    
    for(int i = 0; i < vector->size; i++) {
        maxValueOfVector = fmax(maxValueOfVector, vector->elements[i]);
        minValueOfVector = fmin(minValueOfVector, vector->elements[i]);
    }
    double range = maxValueOfVector - minValueOfVector;

    for(int i = 0; i < vector->size; i++) {
        vector->elements[i] = (vector->elements[i] - minValueOfVector) / range;
    }

}

void unnormalizeVector(Vector* vector, double min, double max) {
    for(int i = 0; i < vector->size; i++) {
        vector->elements[i] = (vector->elements[i] * (max - min)) + min;
    }
}
/*
    @param categories: the column containing the expected category
*/
Matrix* oneHotEncode(Vector* categories, int numberOfCategories) {
 
    // Create a matrix to hold the one-hot encoded categories
    Matrix* oneHotEncoded = create_matrix(categories->size, numberOfCategories);

    // Encode each category value
    for (int i = 0; i < categories->size; i++) {
        double category = categories->elements[i] - 1;

        for (int j = 0; j < numberOfCategories; j++) {
            if (j == category) {
                oneHotEncoded->data[i]->elements[j] = 1.0;
            } else {
                oneHotEncoded->data[i]->elements[j] = 0.0;  
            }
        }
    }
    return oneHotEncoded;
}

void free_data(Data* data) {
    free_matrix(data->data);
    free(data);
}