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
    Matrix* yValues;

    Vector* minValues;
    Vector* maxValues;

    Matrix* trainingOutputs;
} Data;

Data* loadCSV(char* fileLocation, double separationFactor, int shouldNormalize);

void shuffleRows(Matrix* matrix);

void removeResultsFromEvaluationSet(Matrix* evalMatrix, int columnIndex);
Vector* extractYValues(Matrix* trainingMatrix, int columnIndex);
Matrix* oneHotEncode(Vector* categories, int numberOfCategories);

int getRowCount(char* fileLocation);
int getColumnCount(char* fileLocation);

void normalizeColumn(Matrix* matrix, int columnIndex);
void normalizeVector(Vector* vector);

void unnormalizeColumn(Matrix* vector, int columnIndex, double min, double max);

void unnormalizeVector(Vector* vector, double min, double max);

#endif