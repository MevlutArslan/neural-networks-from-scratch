#include "data_processing.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#define MAX_LINE_LENGTH 1024
#define MAX_TOKENS 256
#define DELIMITER ","


// TODO: Generalize this to accept indices of the rows and columns we want to skip
Data* loadCSV(char* fileLocation, double separationFactor, int shouldNormalize) {
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

    data->numberOfColumns = getColumnCount(fileLocation);
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
        while ((token = strtok_r(rest, DELIMITER, &rest)) && rowIndex < data->numberOfRows + 1 && colIndex < data->numberOfColumns) {
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
    shuffleRows(matrix);

    data->yValues = oneHotEncode(extractYValues(matrix, 0), 3);

    data->trainingData = createMatrix(data->numberOfRows * separationFactor, data->numberOfColumns - 1);

    for(int i = 0; i < data->trainingData->rows; i++) {
        for(int j = 0; j < data->numberOfColumns - 1; j++) {
            data->trainingData->data[i]->elements[j] = matrix->data[i]->elements[j+1];
        }
    }
    
    data->evaluationData = createMatrix(data->numberOfRows - data->trainingData->rows, data->numberOfColumns - 1);

    for(int i = 0; i < data->trainingData->rows; i++) {
        for(int j = 0; j < data->numberOfColumns - 1; j++) {
            if(data->trainingData->rows + i < matrix->rows) {
                data->evaluationData->data[i]->elements[j] = matrix->data[data->trainingData->rows + i]->elements[j + 1];
            }
        }
    }

    if(shouldNormalize == 1) {
        Vector* maxValues = createVector(data->trainingData->columns);
        Vector* minValues = createVector(data->trainingData->columns);
        for (int col = 0; col < data->trainingData->columns; col++) {
            maxValues->elements[col] = -DBL_MAX;
            minValues->elements[col] = DBL_MAX;
        }

        for (int row = 0; row < data->numberOfRows; row++) {
            for (int col = 0; col < data->trainingData->columns; col++) {
                double value = matrix->data[row]->elements[col];
            
                if (value > maxValues->elements[col]) {
                    maxValues->elements[col] = value;
                }
                
                if (value < minValues->elements[col]) {
                    minValues->elements[col] = value;
                }
            }
        }
        for(int col = 0; col < data->trainingData->columns; col++) {
            normalizeColumn(data->trainingData, col);
        }
       
        for(int col = 0; col < data->trainingData->columns; col++) {
            normalizeColumn(data->evaluationData, col);
        }


        data->maxValues = maxValues;
        data->minValues = minValues;
    }
    
    freeMatrix(matrix);
    // removeResultsFromEvaluationSet(data->evaluationData, 0);
   
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

void shuffleRows(Matrix* matrix) {
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
    Matrix* shuffledMatrix = createMatrix(numberOfRows, matrix->columns);

    // Fill the new matrix with rows in the order specified by the permutation
    for (int i = 0; i < numberOfRows; i++) {
        shuffledMatrix->data[i] = matrix->data[permutation[i]];
    }

    // Replace the old matrix data with the shuffled data
    free(matrix->data);
    matrix->data = shuffledMatrix->data;

    // Clean up
    free(shuffledMatrix);
    free(permutation);
}

Matrix* oneHotEncode(Vector* categories, int numberOfCategories) {
    // Create a matrix to hold the one-hot encoded categories
    Matrix* oneHotEncoded = createMatrix(categories->size, numberOfCategories);

    // For each category in the input vector
    for (int i = 0; i < categories->size; i++) {
        // Subtract 1 from the category to get a zero-based index
        int index = (int)categories->elements[i] - 1;

        // Set the corresponding element in the one-hot encoded matrix to 1
        oneHotEncoded->data[i]->elements[index] = 1.0;
    }

    return oneHotEncoded;
}