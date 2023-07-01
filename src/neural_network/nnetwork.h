#ifndef NNETWORK_H
#define NNETWORK_H

#include "layer.h"
#include "../nmath/nmatrix.h"
#include "../nmath/nvector.h"
#include "../nmath/nmath.h"
#include "../helper/data_processing.h"
#include "loss_functions.h"

typedef struct {
    Data* data;

    int layerCount;
    Layer** layers;

    double (*loss_function)(Vector*, Vector*);
    Vector* loss;
    double totalLoss;
} NNetwork;

typedef struct {
    int numLayers;               // Number of layers in the network
    int* neuronsPerLayer;        // Array of number of neurons per layer
    ActivationFunction* activationFunctions;  // Array of activation functions for each layer
    double learningRate;         // Learning rate for training the network
    double (*loss_function)(Vector*, Vector*);
    Data* data;
} NetworkConfig;

NNetwork* createNetwork(const NetworkConfig* config);
void deleteNNetwork(NNetwork* network);

void forwardPass(NNetwork* network);
// void backpropagation(NNetwork* network);
// void updateWeightsAndBiases(NNetwork* network, double learningRate);


// void calculateOutputLayerGradient(Layer* outputLayer, Vector* target);
// void calculateHiddenLayerGradient(Layer* layer);

#endif