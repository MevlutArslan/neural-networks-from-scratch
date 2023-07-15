#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include "../nmath/nmath.h"
#include <stdio.h>
#include "../../libraries/logger/log.h"
#include "../helper/constants.h"

typedef struct {
    double (*loss_function)(Matrix*, Matrix*);
    double (*derivative)(double, double);
} LossFunction;

double meanSquaredError(Matrix* outputs, Matrix* targets);
double meanSquaredErrorDerivative(double target, double predicted);

double categoricalCrossEntropyLoss(Matrix* targetOutputs, Matrix* outputs);
double categoricalCrossEntropyPerInput(Vector* target, Vector* output);
Vector* categoricalCrossEntropyLossDerivative(Vector* target, Vector* predicted);

#endif