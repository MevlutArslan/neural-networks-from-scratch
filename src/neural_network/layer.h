#ifndef LAYER_H
#define LAYER_H

#include "activation_function.h"
#include "../nmath/nmath.h"
#include "../helper/constants.h"

typedef struct Layer{
    int neuronCount;
    Vector* input;
    
    Matrix* weights;
    Vector* biases;

    Vector* weightedSums;
    Vector* output;

    // Vector* error;
    Matrix* gradients;
    Vector* biasGradients;

    ActivationFunction* activationFunction;
    Vector* dLoss_dWeightedSums;

    Matrix* weightMomentums;
    Vector* biasMomentums;

    Matrix* weightCache;
    Vector* biasCache;

    // lambda dictates how much of an impact the regularization penalty carries.
    double weightLambda;
    double biasLambda;
} Layer;

typedef struct {
    int neuronCount;
    int inputSize;
    ActivationFunction* activationFunction;
    
    int shouldUseRegularization;
    double weightLambda;
    double biasLambda;
} LayerConfig;

Layer* create_layer(LayerConfig* config);
void free_layer(Layer* layer);

void initialize_weights_he(int inputNeuronCount, int outputNeuronCount, Matrix* weights);
#endif