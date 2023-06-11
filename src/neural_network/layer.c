#include "layer.h"
#include "neuron.h"

Layer* createLayer(int numberOfNeurons, Vector* inputs) {
    Layer* layer = malloc(sizeof(Layer));

    layer->numNeurons = numberOfNeurons;
    layer->neurons = malloc(numberOfNeurons * sizeof(Neuron));
    layer->inputs = inputs;
    layer->outputs = createVector(numberOfNeurons);
    
    for(int i = 0; i < numberOfNeurons; i++) {
        layer->neurons[i] = createNeuron(layer->inputs->size);
    }
    
    return layer;
}

void deleteLayer(Layer* layer) {
    if (layer == NULL) {
        return;
    }

    if(layer->next != NULL) { 
        deleteLayer(layer->next);
    }

    deleteVector(layer->inputs);
    
    for (int i = 0; i < layer->numNeurons; i++) {
        deleteNeuron(layer->neurons[i]);
    }

    free(layer);
}