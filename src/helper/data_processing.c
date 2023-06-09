#include "data_processing.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#define MAX_LINE_LENGTH 1024
#define MAX_TOKENS 256
#define DELIMITER ","

Data* load_csv(char* fileLocation) {
    FILE* file = fopen(fileLocation, "r");
    if (file == NULL) {
        printf("Failed to open file: %s\n", fileLocation);
        return NULL;
    }

    Data* data = malloc(sizeof(Data));
    if (data == NULL) {
        printf("Failed to allocate memory for Data\n");
        fclose(file);
        return NULL;
    }

    data->rows = (getRowCount(fileLocation) - 1);
    data->columns = getColumnCount(fileLocation);
    
    Matrix* matrix = create_matrix(data->rows, data->columns);
    if (matrix == NULL) {
        printf("Failed to create matrix\n");
        fclose(file);
        free(data);
        return NULL;
    }

    char currentLine[MAX_LINE_LENGTH];
    int rowIndex = 0;

    // while there are lines to read
    while (fgets(currentLine, sizeof(currentLine), file) != NULL) {

        if (currentLine[0] == '\n' || currentLine[0] == '\r') {
            continue;
        }

        char* token;
        char* rest = currentLine;
        int colIndex = 0;

        // strtok separates a line by the delimiter and writes the remaining of the line into the address provided
        // we look a step ahead in the while loop but we assign that to the correct index of the matrix
        while ((token = strtok_r(rest, DELIMITER, &rest)) && rowIndex < data->rows + 1 && colIndex < data->columns) {
            if (rowIndex == 0) {
                break;
            }
            
            double value = atof(token);

            matrix->data[rowIndex - 1]->elements[colIndex] = value;

            colIndex++;
        }

        rowIndex++;
    }

    fclose(file);

    data->data = matrix;
   
    return data;
}

Vector* extractYValues(Matrix* matrix, int columnIndex) {
    Vector* yValues = create_vector(matrix->rows);

    for(int i = 0; i < yValues->size; i++) {
        yValues->elements[i] = matrix->data[i]->elements[columnIndex];
        matrix->data[i]->elements[columnIndex] = 0.0f;
    }

    return yValues;
}

int getColumnCount(char* fileLocation) {
    FILE* file = fopen(fileLocation, "r");
    if(file == NULL) {
        printf("Failed to open file: %s\n", fileLocation);
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



int getRowCount(char* fileLocation) {
    FILE* file = fopen(fileLocation, "r");
    if(file == NULL) {
        printf("Failed to open file: %s\n", fileLocation);
        return 0;
    }

    int rowCount = 0;
    char currentLine[MAX_LINE_LENGTH];
    while (fgets(currentLine, sizeof(currentLine), file) != NULL) {
        if(currentLine[0] == '\n' || currentLine[0] == '\r') {
            continue;
        }

        rowCount++;
    }

    fclose(file);
    return rowCount;
}


void normalizeColumn(Matrix* matrix, int columnIndex) {
    double maxValueOfColumn = DBL_MIN;
    double minValueOfColumn = DBL_MAX;

    for(int row = 0; row < matrix->rows; row++) {
        maxValueOfColumn = fmax(maxValueOfColumn, matrix->data[row]->elements[columnIndex]);
        minValueOfColumn = fmin(minValueOfColumn, matrix->data[row]->elements[columnIndex]);
    }

    for(int row = 0; row < matrix->rows; row++) {
        double value = matrix->data[row]->elements[columnIndex];
        matrix->data[row]->elements[columnIndex] = (value - minValueOfColumn) / (maxValueOfColumn - minValueOfColumn);
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

Matrix* oneHotEncode(Vector* categories, int numberOfCategories) {
    // Create a matrix to hold the one-hot encoded categories
    Matrix* oneHotEncoded = create_matrix(categories->size, numberOfCategories);

    // For each category in the input vector
    for (int i = 0; i < categories->size; i++) {
        // Subtract 1 from the category to get a zero-based index
        int index = (int)categories->elements[i] - 1;

        // Set the corresponding element in the one-hot encoded matrix to 1
        oneHotEncoded->data[i]->elements[index] = 1.0;
    }

    return oneHotEncoded;
}