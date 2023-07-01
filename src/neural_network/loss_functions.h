#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include "../nmath/nvector.h"

typedef struct {
    
} LossFunction;

double meanSquaredError(double output, double target);
double derivativeMeanSquaredError(double output, double target);
double calculateAverageLoss(Vector* actual, Vector* predicted);

#endif