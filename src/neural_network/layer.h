#ifndef LAYER_H
#define LAYER_H

#include "../nmath/nvector.h"
#include "neuron.h"
#include "activation_functions/activation_function.h"

typedef struct Layer{
    // A layer is a group of neurons
    // each neural takes equal amount of inputs
    // So I suppose we need inputs but to create inputs we will need the shape of the vector
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