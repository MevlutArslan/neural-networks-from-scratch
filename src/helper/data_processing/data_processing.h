#ifndef DATA_PROCESSING_H
#define DATA_PROCESSING_H

#include <csv.h>
#include "../../nmath/nmath.h"

typedef struct {
    // maybe rename this. i just want to keep track of the names of columns (feature names or something like that)
    char** columnNames;
    int numberOfRows;
    int numberOfColumns;

    Matrix* evaluationData;
    Matrix* trainingData;
    Vector* yValues;
    Vector* outputs;
} Data;

Data* loadCSV(char* fileLocation, double separationFactor);

void removeResultsFromEvaluationSet(Matrix* evalMatrix, int columnIndex);
Vector* extractYValues(Matrix* trainingMatrix, int columnIndex);

int getRowCount(char* fileLocation);
int getColumnCount(char* fileLocation);

#endif