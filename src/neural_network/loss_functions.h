#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include "../nmath/nmath.h"
#include <stdio.h>

typedef struct {
    double (*loss_function)(Matrix*, Vector*);
    double (*derivative)(double);
} LossFunction;

double meanSquaredError(Matrix* outputs, Vector* targets);
// double meanSquaredErrorDerivative(Matrix* outputs, Vector* target);


#endif