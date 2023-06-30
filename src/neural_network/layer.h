#ifndef LAYER_H
#define LAYER_H

#include "../nmath/nvector.h"
#include "activation_functions/activation_function.h"
#include "../nmath/nmath.h"

typedef struct Layer{
    int neuronCount;
    Matrix* input;
    Matrix* weights;
    Vector* biases;
    Vector* output;

    ActivationFunction* activationFunction;
    OutputActivationFunction* outputActivationFunction;
    struct Layer* prev; // for backward propogation
    struct Layer* next;
} Layer;

typedef struct {
    int neuronCount;
    Vector* input;

    ActivationFunction* activationFunction;
    OutputActivationFunction* outputActivationFunction;
} LayerConfig;

Layer* createLayer(LayerConfig* config);
void deleteLayer(Layer* layer);

#endif