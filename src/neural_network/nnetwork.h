#ifndef NNETWORK_H
#define NNETWORK_H

#include "layer.h"
#include "../nmath/nmatrix.h"
#include "../nmath/nvector.h"
#include "../nmath/nmath.h"
#include "../helper/data_processing.h"
#include "loss_functions.h"

typedef struct {
    int shouldUseGradientClipping;
    double gradientClippingLowerBound;
    double gradientClippingUpperBound;

    int shouldUseLearningRateDecay;
    double learningRateDecayAmount;

    int shouldUseMomentum;
    double momentum;

} OptimizationConfig;

typedef struct {
    Data* data;

    int layerCount;
    Layer** layers;

    LossFunction* lossFunction;
    double loss;

    OptimizationConfig* optimizationConfig;
} NNetwork;

typedef struct {
    int numLayers;               // Number of layers in the network
    int* neuronsPerLayer;        // Array of number of neurons per layer
    ActivationFunction* activationFunctions;  // Array of activation functions for each layer
    double learningRate;         // Learning rate for training the network
    LossFunction* lossFunction;
    Data* data;

    OptimizationConfig* optimizationConfig;
} NetworkConfig;

NNetwork* createNetwork(const NetworkConfig* config);
void deleteNNetwork(NNetwork* network);

void forwardPass(NNetwork* network);
void backpropagation(NNetwork* network);
void optimize(NNetwork* network, double learningRate);

void dumpNetworkState(NNetwork* network);
#endif