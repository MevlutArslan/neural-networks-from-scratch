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

    Vector* weightLambdas;
    Vector* biasLambdas;
} NetworkConfig;

NNetwork* create_network(const NetworkConfig* config);
void free_network(NNetwork* network);
void free_network_config(NetworkConfig* config);

void forward_pass(NNetwork* network, Vector* input, Vector* output);
void calculate_loss(NNetwork* network, Matrix* yValues);
void backpropagation(NNetwork* network, Vector* input, Vector* output, Vector* yValues);

void dump_network_config(NNetwork* network);

void sgd(NNetwork* network, double learningRate);
void adagrad(NNetwork* network, double learningRate);
void rms_prop(NNetwork* network, double learningRate);
void adam(NNetwork* network, double learningRate);

/*
    L1 regularization’s penalty is the sum of all the absolute values for the weights and biases.
    weights_penalty = lambda * sum(weights)
    bias_penalty = lambda * sum(biases)

    @param lambda dictates how much of an impact the regularization penalty carries.
*/
double calculate_l1_penalty(double lambda, const Vector* vector);
Vector* l1_derivative(double lambda, const Vector* vector);

/*
    L2 regularization’s penalty is the sum of the squared weights and biases.
    updated_weight = lambda * sum(weights^2)
    updated_bias = lambda * sum(biases ^ 2)

    @param lambda variable dictates how much of an impact the regularization penalty carries.
*/
double calculate_l2_penalty(double lambda, const Vector* vector);
Vector* l2_derivative(double lambda, const Vector* vector);

double accuracy(Matrix* targets, Matrix* outputs);
#endif