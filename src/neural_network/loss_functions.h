#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include "../nmath/nmath.h"
#include <stdio.h>
#include "../../libraries/logger/log.h"
#include "../helper/constants.h"

typedef struct {
    double (*loss_function)(Matrix*, Matrix*);
    double (*derivative)(double, double);
    const char* name;
} LossFunction;

extern LossFunction MEAN_SQUARED_ERROR;
extern LossFunction CATEGORICAL_CROSS_ENTROPY;

double meanSquaredError(Matrix* outputs, Matrix* targets);
double meanSquaredErrorDerivative(double target, double predicted);

double categoricalCrossEntropyLoss(Matrix* targetOutputs, Matrix* outputs);
double categoricalCrossEntropyPerInput(Matrix* target, int target_index, Matrix* output, int output_index);
Vector* categoricalCrossEntropyLossDerivative(Vector* target, Vector* predicted);
void computeCategoricalCrossEntropyLossDerivativeMatrix(Matrix* target, Matrix* prediction, Matrix* loss_wrt_output);

const char* get_loss_function_name(const LossFunction* lossFunction);
#endif