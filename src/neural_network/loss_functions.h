#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include "../nmath/nmath.h"
#include <stdio.h>

double meanSquaredError(Matrix* outputs, Vector* targets);
double derivativeMeanSquaredError(double output, double target);


#endif