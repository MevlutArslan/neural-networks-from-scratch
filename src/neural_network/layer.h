#ifndef LAYER_H
#define LAYER_H

#include "activation_function.h"
#include "../nmath/nmath.h"
#include "../helper/constants.h"

typedef struct Layer{
    int num_neurons;
    Vector* input;
    
    Matrix* weights;
    Vector* biases;

    Vector* weighted_sums;
    Vector* output;

    Matrix* weight_gradients;
    Vector* bias_gradients;

    ActivationFunction activation_fn;
    Vector* loss_wrt_wsums;

    Matrix* weight_momentums;
    Vector* bias_momentums;
    
    Matrix* weight_cache;
    Vector* bias_cache;

    // lambda dictates how much of an impact the regularization penalty carries.
    float l1_weight_lambda;
    float l1_bias_lambda;

    float l2_weight_lambda;
    float l2_bias_lambda;
} Layer;

typedef struct {
    int num_neurons;
    int num_inputs;
    ActivationFunction activation_fn;
    
    int use_l1_regularization;
    int use_l2_regularization;

    float l1_weight_lambda;
    float l1_bias_lambda;

    float l2_weight_lambda;
    float l2_bias_lambda;
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