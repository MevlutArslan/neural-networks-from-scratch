#ifndef LAYER_H
#define LAYER_H

#include "activation_function.h"
#include "../nmath/nmath.h"

typedef struct Layer{
    int neuronCount;
    Matrix* input;
    Matrix* weights;
    Vector* biases;
    Vector* output;

    ActivationFunction* activationFunction;
    struct Layer* prev; // for backward propogation
    struct Layer* next;
} Layer;

typedef struct {
    int neuronCount;
    int inputSize;
    ActivationFunction* activationFunction;
} LayerConfig;

Layer* createLayer(LayerConfig* config);
void deleteLayer(Layer* layer);

#endif