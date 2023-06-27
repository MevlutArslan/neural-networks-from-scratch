#ifndef NNETWORK_H
#define NNETWORK_H

#include "layer.h"
#include "../nmath/nmath.h"
#include "../helper/data_processing/data_processing.h"

typedef struct {
    Data* data;
    Layer* start;
    Layer* end;
    double (*loss_function)(Vector*, Vector*);
} NNetwork;

typedef struct {
    int numLayers;               // Number of layers in the network
    int* neuronsPerLayer;        // Array of number of neurons per layer
    ActivationFunction* activationFunctions;  // Array of activation functions for each layer
    OutputActivationFunction* outputActivationFunction;
    double learningRate;         // Learning rate for training the network
    double (*loss_function)(Vector*, Vector*);
} NetworkConfig;

NNetwork* createNetwork(const NetworkConfig* config, Vector* input);
void deleteNNetwork(NNetwork* network);

void forwardPass(NNetwork* network);
void backpropagation(NNetwork* network);
void updateWeightsAndBiases(NNetwork* network, double learningRate);


void calculateOutputLayerGradient(Layer* outputLayer, Vector* target);
void calculateHiddenLayerGradient(Layer* layer);

#endif