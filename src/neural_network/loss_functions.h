#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include "../nmath/nmath.h"
#include <stdio.h>
#include "../../libraries/logger/log.h"
#include "../helper/constants.h"

#define MEAN_SQUARED_ERROR_STR "mean_squared_error"
#define CATEGORICAL_CROSS_ENTROPY_STR "categorical_cross_entropy"
typedef enum LossFunctionType  {
    MEAN_SQUARED_ERROR, CATEGORICAL_CROSS_ENTROPY, UNRECOGNIZED_LFN
} LossFunctionType;

double mean_squared_error(Matrix* targets, Matrix* outputs);
double mean_squared_error_derivative(double target, double predicted);

double categorical_cross_entropy_loss(Matrix* targetOutputs, Matrix* outputs);
double categorical_cross_entropy_loss_per_row(Vector* target, Vector* output);

Vector* categorical_cross_entropy_loss_derivative(Vector* target, Vector* predicted);
void categorical_cross_entropy_loss_derivative_batched(Matrix* target, Matrix* prediction, Matrix* loss_wrt_output);

LossFunctionType  get_loss_fn_by_name(char* name);
const char* loss_fn_to_string(const LossFunctionType  lossFunction);

#endif