#ifndef LAYER_H
#define LAYER_H

#include "activation_function.h"
#include "../nmath/nmath.h"
#include "../helper/constants.h"

typedef struct Layer{
    int neuronCount;
    Vector* input;
    Matrix* weights;
    Vector* biases;

    Vector* weightedSums;
    Vector* output;

    Vector* error;
    Matrix* gradients;
    Vector* biasGradients;

    ActivationFunction* activationFunction;
    Vector* dLoss_dWeightedSums;

    Matrix* weightMomentums;
    Vector* biasMomentums;

    Matrix* weightCache;
    Vector* biasCache;
} Layer;

typedef struct {
    int neuronCount;
    int inputSize;
    ActivationFunction* activationFunction;

    int willUseMomentum;
    int optimizer;
} LayerConfig;

Layer* createLayer(LayerConfig* config);
void deleteLayer(Layer* layer);

#endif