#ifndef LAYER_H
#define LAYER_H

#include "activation_function.h"
#include "../nmath/nmath.h"

typedef struct Layer{
    int neuronCount;
    Vector* input;
    Matrix* weights;
    Vector* biases;

    Vector* weightedSums;
    Vector* output;

    Vector* error;
    Matrix* gradients;

    ActivationFunction* activationFunction;
    Vector* dLoss_dWeightedSums;
} Layer;

typedef struct {
    int neuronCount;
    int inputSize;
    ActivationFunction* activationFunction;
} LayerConfig;

Layer* createLayer(LayerConfig* config);
void deleteLayer(Layer* layer);

#endif