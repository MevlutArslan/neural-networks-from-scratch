#ifndef LAYER_H
#define LAYER_H

#include "activation_function.h"
#include "../nmath/nmath.h"
#include "../helper/constants.h"

typedef struct Layer{
    int neuronCount;
    Matrix* input; // <- not threadsafe
    
    Matrix* weights;
    Vector* biases;

    Matrix* weightedSums; // <- not threadsafe
    Matrix* output; // <- not threadsafe

    // Vector* error;
    Matrix* gradients; // <- not threadsafe
    Vector* biasGradients; // <- not threadsafe

    ActivationFunction* activationFunction;
    Vector* dLoss_dWeightedSums; // <- not threadsafe

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

/*
 * Serializes a Layer struct into a JSON string.
 *
 * NOTE: Only call this function after the network has been trained.
 * Many of the matrices within the Layer are not initialized at the start,
 * so calling this function prematurely can lead to segmentation faults.
 */
char* serialize_layer(const Layer* layer);
Layer* deserialize_layer(cJSON* json);
#endif