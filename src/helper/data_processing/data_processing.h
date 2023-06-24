#ifndef DATA_PROCESSING_H
#define DATA_PROCESSING_H

#include <csv.h>
#include "../../nmath/nmath.h"

typedef struct {
    // maybe rename this. i just want to keep track of the names of columns (feature names or something like that)
    char** columnNames;
    int numberOfRows;
    int numberOfColumns;
    
    Matrix* matrix;
} Data;

// I haven't decided in what format to load
Data* loadCSV(char* fileLocation);

int getRowCount(char* fileLocation);
int getColumnCount(char* fileLocation);

#endif