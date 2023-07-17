#ifndef DATA_PROCESSING_H
#define DATA_PROCESSING_H

#include <csv.h>
#include "../nmath/nmath.h"

// TODO: Move data separation out of this
typedef struct {
    int rows;
    int columns;
    Matrix* data;
} Data;

Data* load_csv(char* fileLocation);

void removeResultsFromEvaluationSet(Matrix* evalMatrix, int columnIndex);
Vector* extractYValues(Matrix* trainingMatrix, int columnIndex);
Matrix* oneHotEncode(Vector* categories, int numberOfCategories);

int getRowCount(char* fileLocation);
int getColumnCount(char* fileLocation);

void normalizeColumn_standard_deviation(Matrix* matrix, int columnIndex);
void normalizeColumn_division(Matrix* matrix, int columnIndex, double toDivideBy);

void normalizeVector(Vector* vector);

void unnormalizeColumn(Matrix* vector, int columnIndex, double min, double max);

void unnormalizeVector(Vector* vector, double min, double max);

void free_data(Data* data);

#endif