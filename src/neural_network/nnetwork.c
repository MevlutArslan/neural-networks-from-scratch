#include "nnetwork.h"
#include "layer.h"

NNetwork* createNetwork(const NetworkConfig* config, Vector* inputs) {
    NNetwork* network = malloc(sizeof(NNetwork));

    Layer* prevLayer = NULL;
    for (int i = 0; i < config->numLayers; i++) {
        Layer* layer = createLayer(config->neuronsPerLayer[i], prevLayer != NULL ? prevLayer->outputs : inputs);
        // layer->activationFunction = config->activationFunctions[i];
        
        if (prevLayer != NULL) {
            prevLayer->next = layer;
        }
     
        if (i == 0) {
            network->start = layer;
        }
        if (i == config->numLayers - 1) {
            network->end = layer;
        }
    
        prevLayer = layer;
        
    }

    return network;
}

void deleteNNetwork(NNetwork* network){
    deleteLayer(network->start);
}

void forwardPass(NNetwork* network) {
    Layer* layer = network->start;

    while(layer != NULL) {
        // for each neuron in the layer
            // calculate dot product of inputs vector of layer and weights vector of the neuron
            // add bias of the neuron
            // place it on the corresponding output
        int counter = 0; 
        for(int i = 0; i < layer->numNeurons; i++) {
            Neuron* neuron = layer->neurons[i];
            for(int k = 0; k < neuron->weights->size; k++) {
                counter++;
            }

            layer->outputs->elements[i] = vector_dot_product(layer->inputs, neuron->weights) + neuron->bias;
        }


        layer = layer->next;
    }
}