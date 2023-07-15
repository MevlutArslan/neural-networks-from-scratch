#ifndef NNETWORK_H
#define NNETWORK_H

#include "layer.h"
#include "../nmath/nmatrix.h"
#include "../nmath/nvector.h"
#include "../nmath/nmath.h"
#include "../helper/data_processing.h"
#include "loss_functions.h"
#include "../helper/constants.h"
#include "../../libraries/logger/log.h"

typedef struct {
    int shouldUseGradientClipping;
    double gradientClippingLowerBound;
    double gradientClippingUpperBound;

    int shouldUseLearningRateDecay;
    double learningRateDecayAmount;

    int shouldUseMomentum;
    double momentum;

    int optimizer;
    
    double epsilon;
    double rho;

    // ADAM
    double beta1;
    double beta2;
} OptimizationConfig;


typedef struct {
    int layerCount;
    Layer** layers;

    LossFunction* lossFunction;
    double loss;
    double accuracy;
    
    void (*optimizer)(struct NNetwork*, double);
    OptimizationConfig* optimizationConfig;

    int currentStep;

    Matrix* output;
} NNetwork;

typedef struct {
    int numLayers;               // Number of layers in the network
    int* neuronsPerLayer;        // Array of number of neurons per layer
    ActivationFunction* activationFunctions;  // Array of activation functions for each layer
    double learningRate;         // Learning rate for training the network
    LossFunction* lossFunction;
    int inputSize;

    OptimizationConfig* optimizationConfig;
} NetworkConfig;

NNetwork* createNetwork(const NetworkConfig* config);
void deleteNNetwork(NNetwork* network);

void forwardPass(NNetwork* network, Matrix* input);
void calculateLoss(NNetwork* network, Matrix* yValues);
void backpropagation(NNetwork* network, Matrix* yValues);

void dumpNetworkState(NNetwork* network);

void sgd(NNetwork* network, double learningRate);
void adagrad(NNetwork* network, double learningRate);
void rms_prop(NNetwork* network, double learningRate);
void adam(NNetwork* network, double learningRate);

double accuracy(Matrix* targets, Matrix* outputs);
#endif