#ifndef LAYER_H
#define LAYER_H

#include "activation_function.h"
#include "../nmath/nmath.h"
#include "../helper/constants.h"

typedef struct Layer{
    int num_neurons;

    Matrix* weights;
    Vector* biases;

    ActivationFunction activation_fn;

    Matrix* weight_momentums;
    Vector* bias_momentums;
    
    Matrix* weight_cache;
    Vector* bias_cache;

    // lambda dictates how much of an impact the regularization penalty carries.
    double weight_lambda;
    double bias_lambda;
} Layer;

typedef struct {
    int num_neurons;
    int num_inputs;
    ActivationFunction activation_fn;
    
    int use_regularization;
    double weight_lambda;
    double bias_lambda;
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