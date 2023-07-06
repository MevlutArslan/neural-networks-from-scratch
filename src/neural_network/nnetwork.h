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

    LossFunction* lossFunction;
    double loss;

    // Gradient Clipping
    int shouldUseGradientClipping;
    int gradientClippingLowerBound;
    int gradientClippingUpperBound;


} NNetwork;

typedef struct {
    int numLayers;               // Number of layers in the network
    int* neuronsPerLayer;        // Array of number of neurons per layer
    ActivationFunction* activationFunctions;  // Array of activation functions for each layer
    double learningRate;         // Learning rate for training the network
    LossFunction* lossFunction;
    Data* data;

    int shouldUseGradientClipping;
    double gradientClippingLowerBound;
    double gradientClippingUpperBound;
} NetworkConfig;

NNetwork* createNetwork(const NetworkConfig* config);
void deleteNNetwork(NNetwork* network);

void forwardPass(NNetwork* network);
void backpropagation(NNetwork* network);
void updateWeightsAndBiases(NNetwork* network, double learningRate);
void calculateNumericalGradients(NNetwork* network, double epsilon);

void dumpNetworkState(NNetwork* network);

// void calculateOutputLayerGradient(Layer* outputLayer, Vector* target);
// void calculateHiddenLayerGradient(Layer* layer);

#endif