#ifndef NNETWORK_H
#define NNETWORK_H

#include "layer.h"
#include "../nmath/nmath.h"

typedef struct {
    Layer* start;
    Layer* end;
} NNetwork;

typedef struct {
    int numLayers;               // Number of layers in the network
    int* neuronsPerLayer;        // Array of number of neurons per layer
    ActivationFunction* activationFunctions;  // Array of activation functions for each layer
    OutputActivationFunction* outputActivationFunction;
    double learningRate;         // Learning rate for training the network
} NetworkConfig;

NNetwork* createNetwork(const NetworkConfig* config, Vector* inputs);
void deleteNNetwork(NNetwork* network);

void forwardPass(NNetwork* network);

#endif