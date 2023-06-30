#include "layer.h"

Layer* createLayer(LayerConfig* config) {
    Layer* layer = malloc(sizeof(Layer));

    layer->neuronCount = config->neuronCount;
    layer->input = reshapeVectorToMatrix(config->input);
    layer->weights = createMatrix(config->input->size, config->neuronCount);
    initializeMatrixWithRandomValuesInRange(layer->weights, -1, 1);
    layer->biases = createVector(config->neuronCount);
    fillVector(layer->biases, 1.0);

    layer->output = createVector(config->neuronCount);
    
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
    free(layer->outputActivationFunction);
    
    free(layer);
}
