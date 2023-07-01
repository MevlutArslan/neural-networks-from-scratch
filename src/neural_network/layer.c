#include "layer.h"

Layer* createLayer(LayerConfig* config) {
    Layer* layer = malloc(sizeof(Layer));

    layer->neuronCount = config->neuronCount;

    layer->weights = createMatrix(config->inputSize, config->neuronCount);
    initializeMatrixWithRandomValuesInRange(layer->weights, 0, 1);

    layer->biases = createVector(config->neuronCount);
    initializeVectorWithRandomValuesInRange(layer->biases, 0, 1.0);

    layer->output = createVector(config->neuronCount);

    layer->activationFunction = config->activationFunction;
    
    return layer;
}

void deleteLayer(Layer* layer) {
    if (layer == NULL) {
        return;
    }

    // Disconnect this layer from any next layer
    if(layer->next != NULL) { 
        layer->next->prev = layer->prev;
    }

    // Disconnect this layer from any previous layer
    if(layer->prev != NULL) {
        layer->prev->next = layer->next;
    }

    // Now it's safe to delete this layer
    freeMatrix(layer->input);
    freeMatrix(layer->weights);
    freeVector(layer->biases);
    freeVector(layer->output);

    free(layer->activationFunction);
    
    free(layer);
}
