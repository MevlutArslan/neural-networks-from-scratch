#ifndef DATA_PROCESSING_H
#define DATA_PROCESSING_H

#include <csv.h>
#include "../nmath/nmath.h"

typedef struct {
    char** columnNames;
    int numberOfRows;
    int numberOfColumns;

    Matrix* evaluationData;
    Matrix* trainingData;
    
    Vector* yValues;
    Matrix* trainingOutputs;
} Data;

Data* loadCSV(char* fileLocation, double separationFactor);

void removeResultsFromEvaluationSet(Matrix* evalMatrix, int columnIndex);
Vector* extractYValues(Matrix* trainingMatrix, int columnIndex);

int getRowCount(char* fileLocation);
int getColumnCount(char* fileLocation);

#endif