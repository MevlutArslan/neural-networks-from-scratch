#include "layer.h"

Layer* create_layer(LayerConfig* config) {
    Layer* layer = malloc(sizeof(Layer));

    layer->neuronCount = config->neuronCount;

    // If 2 subsequent layers have X and Y neurons, then the number of weights is X*Y
    layer->weights = create_matrix(config->neuronCount, config->inputSize);
    initialize_weights_he(config->inputSize, config->neuronCount, layer->weights);
    
    layer->gradients = create_matrix(layer->weights->rows, layer->weights->columns);
    fill_matrix(layer->gradients, 0.0);

    layer->biases = create_vector(config->neuronCount);
    fill_vector_random(layer->biases, -0.5, 0.5);

    layer->biasGradients = create_vector(config->neuronCount);
    fill_vector(layer->biases, 0.0f);

    layer->weightedSums = create_vector(config->neuronCount);
    layer->output = create_vector(config->neuronCount);

    layer->activationFunction = config->activationFunction;

    layer->dLoss_dWeightedSums = create_vector(layer->neuronCount);
    
    layer->weightMomentums = create_matrix(layer->weights->rows, layer->weights->columns);
    layer->biasMomentums = create_vector(layer->biases->size);
    layer->weightCache = create_matrix(layer->weights->rows, layer->weights->columns);
    layer->biasCache = create_vector(layer->biases->size);
    
    if(config->shouldUseRegularization == 1) {
        if(config->weightLambda != 0) {
            layer->weightLambda = config->weightLambda;
        }
        if(config->biasLambda != 0) {
            layer->biasLambda = config->biasLambda;
        }
    }
    return layer;
}

void free_layer(Layer* layer) {
    if (layer == NULL) {
        return;
    }

    // Free the resources allocated for the layer
    free_matrix(layer->input);
    free_matrix(layer->weights);
    free_vector(layer->biases);
    free_vector(layer->output);
    free_vector(layer->error);
    free_matrix(layer->gradients);
    free_vector(layer->biasGradients);
    free_vector(layer->dLoss_dWeightedSums);
    free_matrix(layer->weightMomentums);
    free_vector(layer->biasMomentums);
    free_matrix(layer->weightCache);
    free_vector(layer->biasCache);

    // Free the layer itself
    free(layer);
}

void initialize_weights_he(int inputNeuronCount, int outputNeuronCount, Matrix* weights) {
    // Calculate limit
    double limit = sqrt(2.0 / (double)inputNeuronCount);

    // Initialize weights
    for(int i = 0; i < outputNeuronCount; i++) {
        for(int j = 0; j < inputNeuronCount; j++) {
            // Generate a random number between -limit and limit
            double rand_num = (double)rand() / RAND_MAX; // This generates a random number between 0 and 1
            rand_num = rand_num * 2 * limit - limit; // This shifts the range to [-limit, limit]
            weights->data[i]->elements[j] = rand_num;
        }
    }
}
