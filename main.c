#include <stdio.h>
#include "math/nmath.h"

int main(void)
{
    // C has char, int, floating point types (float, double)
    // no boolean in C
    // use 1 for true and 0 for false
    Matrix* matrix = createMatrix(2,2);
    
    printf("Hello World! \n");
    printMatrix(matrix);

    freeMatrix(matrix);
}