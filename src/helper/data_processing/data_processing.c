#include "data_processing.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 1024
#define MAX_TOKENS 256
#define DELIMITER ","

Data* loadCSV(char* fileLocation) {
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
    data->matrix = createMatrix(data->numberOfRows, data->numberOfColumns);
    if (data->matrix == NULL) {
        printf("Failed to create matrix\n");
        fclose(file);
        free(data);
        return NULL;
    }

    char currentLine[MAX_LINE_LENGTH];
    int rowIndex = 0;
    int tokenIndex = 0;

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

            // want to skip first col or first row cause they are useless
            data->matrix->data[rowIndex - 1]->elements[colIndex - 1] = value;

            colIndex++;
        }

        rowIndex++;
    }

    fclose(file);
    return data;
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
