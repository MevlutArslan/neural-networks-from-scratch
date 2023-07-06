#include "layer.h"

Layer* createLayer(LayerConfig* config) {
    Layer* layer = malloc(sizeof(Layer));

    layer->neuronCount = config->neuronCount;

    // If 2 subsequent layers have X and Y neurons, then the number of weights is X*Y
    layer->weights = createMatrix(config->neuronCount, config->inputSize);
    initializeMatrixWithRandomValuesInRange(layer->weights, -1, 1);

    layer->biases = createVector(config->neuronCount);
    initializeVectorWithRandomValuesInRange(layer->biases, -0.5, 0.5);

    layer->weightedSums = createVector(config->neuronCount);
    layer->output = createVector(config->neuronCount);

    layer->activationFunction = config->activationFunction;
    
    layer->gradients = createMatrix(layer->weights->rows, layer->weights->columns);
    layer->dLoss_dWeightedSums = createVector(layer->neuronCount);
    return layer;
}

void deleteLayer(Layer* layer) {
    if (layer == NULL) {
        return;
    }

    // Now it's safe to delete this layer
    freeMatrix(layer->input);
    freeMatrix(layer->weights);
    freeVector(layer->biases);
    freeVector(layer->output);

    free(layer->activationFunction);
    
    free(layer);
}
