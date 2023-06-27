#ifndef LAYER_H
#define LAYER_H

#include "../nmath/nvector.h"
#include "neuron.h"
#include "activation_functions/activation_function.h"

typedef struct Layer{
    int numNeurons;
    Vector* inputs;
    Neuron** neurons;
    Vector* outputs;
    ActivationFunction* activationFunction;
    OutputActivationFunction* outputActivationFunction;
    struct Layer* prev; // for backward propogation
    struct Layer* next;
} Layer;

Layer* createLayer(int numberOfNeurons, Vector* inputs);
void deleteLayer(Layer* layer);


#endif