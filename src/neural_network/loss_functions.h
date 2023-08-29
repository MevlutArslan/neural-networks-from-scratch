#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include "../nmath/nmath.h"
#include "../../libraries/logger/log.h"
#include "../helper/constants.h"

#define MEAN_SQUARED_ERROR_STR "mean_squared_error"
#define CATEGORICAL_CROSS_ENTROPY_STR "categorical_cross_entropy"
typedef enum LossFunctionType  {
    MEAN_SQUARED_ERROR, CATEGORICAL_CROSS_ENTROPY, UNRECOGNIZED_LFN
} LossFunctionType ;

double calculateMeanSquaredError(Matrix* predicted, Matrix* target);
double calculateMeanSquaredErrorDerivative(double target, double predicted);

double calculateCategoricalCrossEntropyLoss(Matrix* predicted, Matrix* target);
double calculateCategoricalCrossEntropyPerInput(Matrix* predicted, int predictedIndex, Matrix* target, int targetIndex);
Vector* calculateCategoricalCrossEntropyLossDerivative(Vector* predicted, Vector* target);

void computeCategoricalCrossEntropyLossDerivativeMatrix(Matrix* target, Matrix* prediction, Matrix* loss_wrt_output);

LossFunctionType  get_loss_fn_by_name(char* name);
const char* loss_fn_to_string(const LossFunctionType  lossFunction);
#endif