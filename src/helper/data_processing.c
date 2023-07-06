#include "data_processing.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#define MAX_LINE_LENGTH 1024
#define MAX_TOKENS 256
#define DELIMITER ","


// TODO: Generalize this to accept indices of the rows and columns we want to skip
Data* loadCSV(char* fileLocation, double separationFactor) {
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

    data->numberOfColumns = getColumnCount(fileLocation) - 1;
    data->numberOfRows = getRowCount(fileLocation) - 1;
    
    Matrix* matrix = createMatrix(data->numberOfRows, data->numberOfColumns);
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
        while ((token = strtok_r(rest, DELIMITER, &rest)) && rowIndex < data->numberOfRows + 1 && colIndex < data->numberOfColumns + 1) {
            if (rowIndex == 0) {
                break;
            }

            if (colIndex == 0) {
                colIndex++;
                continue;
            }

            double value = atof(token);

            // want to skip first col and first row cause they are useless
            matrix->data[rowIndex - 1]->elements[colIndex - 1] = value;

            colIndex++;
        }

        rowIndex++;
    }

    fclose(file);

    data->trainingData = createMatrix(data->numberOfRows * separationFactor, data->numberOfColumns);

    for(int i = 0; i < data->trainingData->rows; i++) {
        for(int j = 0; j < data->numberOfColumns; j++) {
            data->trainingData->data[i]->elements[j] = matrix->data[i]->elements[j];
        }
    }

    for(int col = 0; col < data->numberOfColumns; col++) {
        normalizeColumn(data->trainingData, col);
    }

    data->yValues = extractYValues(matrix, data->numberOfColumns - 1);
    normalizeVector(data->yValues);
    
    data->evaluationData = createMatrix(data->numberOfRows - data->trainingData->rows, data->numberOfColumns);

    for(int i = 0; i < data->trainingData->rows; i++) {
        for(int j = 0; j < data->numberOfColumns; j++) {
            if(data->trainingData->rows + i < matrix->rows) {
                data->evaluationData->data[i]->elements[j] = matrix->data[data->trainingData->rows + i]->elements[j];
            }
        }
    }

    for(int col = 0; col < data->numberOfColumns; col++) {
        normalizeColumn(data->evaluationData, col);
    }

    freeMatrix(matrix);
    removeResultsFromEvaluationSet(data->evaluationData, data->numberOfColumns - 1);

    return data;
}

void removeResultsFromEvaluationSet(Matrix* evalMatrix, int columnIndex) {
    for(int i = 0; i < evalMatrix->rows; i++) {
        evalMatrix->data[i]->elements[columnIndex] = 0.0f;
    }
}

Vector* extractYValues(Matrix* matrix, int columnIndex) {
    Vector* yValues = createVector(matrix->rows);

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

    for(int i = 0; i < vector->size; i++) {
        vector->elements[i] = (vector->elements[i] - minValueOfVector) / maxValueOfVector;
    }
}