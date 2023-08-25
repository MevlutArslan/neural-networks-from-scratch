#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include "../nmath/nmath.h"
#include "../../libraries/logger/log.h"
#include "../helper/constants.h"

#define MEAN_SQUARED_ERROR_STR "mean_squared_error"
#define CATEGORICAL_CROSS_ENTROPY_STR "categorical_cross_entropy"
typedef enum LossFunction {
    MEAN_SQUARED_ERROR, CATEGORICAL_CROSS_ENTROPY, UNRECOGNIZED_LFN
} LossFunction;

double meanSquaredError(Matrix* outputs, Matrix* targets);
double meanSquaredErrorDerivative(double target, double predicted);

double categoricalCrossEntropyLoss(Matrix* targetOutputs, Matrix* outputs);
double categoricalCrossEntropyPerInput(Matrix* target, int target_index, Matrix* output, int output_index);
Vector* categoricalCrossEntropyLossDerivative(Vector* target, Vector* predicted);
void computeCategoricalCrossEntropyLossDerivativeMatrix(Matrix* target, Matrix* prediction, Matrix* loss_wrt_output);

LossFunction get_loss_fn_by_name(char* name);
const char* loss_fn_to_string(const LossFunction lossFunction);
#endif