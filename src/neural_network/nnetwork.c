#include "nnetwork.h"
#include "layer.h"

/*
    createNetwork works by getting the config and the inputs.
    It allocates memory for the Neural Network based on the struct 

    typedef struct {
        Layer* start;
        Layer* end;
    } NNetwork;

    Where each layer holds information like :
    * number of neurons
    * pointer to a vector of inputs
    * Pointer to an array of Neurons
    * Pointer to a vector of outputs
    * Pointer to an activation function
    * Pointer to the next layer (Layers can classified as linked lists)
    
    we loop config.number of layers times and create each layer.
    we mark the first layer to be network's start layer and the last layer to be the end layer
*/
NNetwork* createNetwork(const NetworkConfig* config, Vector* input) {
    NNetwork* network = malloc(sizeof(NNetwork));

    Layer* prevLayer = NULL;
    for (int i = 0; i < config->numLayers; i++) {
        Layer* layer = createLayer(config->neuronsPerLayer[i], prevLayer != NULL ? prevLayer->outputs : input);
        layer->activationFunction = &config->activationFunctions[i];    

        if (prevLayer != NULL) {
            prevLayer->next = layer;
        }
     
        if (i == 0) {
            network->start = layer;
        }
        if (i == config->numLayers - 1) {
            layer->outputActivationFunction = config->outputActivationFunction;
            network->end = layer;
        }else {
            layer->outputActivationFunction = NULL;
        }
    
        prevLayer = layer;
    }

    network->loss_function = config->loss_function;

    return network;
}

void deleteNNetwork(NNetwork* network){
    deleteLayer(network->start);
}

/*
    For each layer in the network:
    1. Iterate over numNeurons per layer.
    2. For each neuron:
        a. Calculate the dot product of the input vector and the neuron's weights.
        b. Add the neuron's bias to the result.
        c. Store the result in the corresponding element of the layer's output vector.
    3. If this is the output layer (i.e., layer->outputActivationFunction is not NULL):
        a. Apply the output activation function (softmax) to the entire output vector.
    4. If this is not the output layer:
        a. Apply the regular activation function (ReLU or sigmoid) to each element of the output vector.
    5. Move to the next layer.
*/
void forwardPass(NNetwork* network) {
    Layer* layer = network->start;

    Vector* outputs = createVector(network->data->trainingData->rows);

    for(int i = 0; i < network->data->trainingData->rows; i++) {
        while(layer != NULL) {

            for(int j = 0; j < layer->numNeurons; j++) {
                Neuron* neuron = layer->neurons[j];
                
                layer->outputs->elements[j] = vector_dot_product(layer->inputs, neuron->weights) + neuron->bias;
            }

            if(layer->outputActivationFunction != NULL) {
                layer->outputActivationFunction->activation(layer->outputs);
            }else {
                for (int j = 0; j < layer->numNeurons; j++) {
                    layer->outputs->elements[j] = layer->activationFunction->activation(layer->outputs->elements[j]);
                }
            }
            
            layer = layer->next;
        }

        for (int i = 0; i < network->end->numNeurons; i++) {
            outputs->elements[i] = network->end->outputs->elements[i];
        }
    }
    network->data->outputs = outputs;

    printf("NETWORK LOSS: %f \n", network->loss_function(network->data->outputs, network->data->yHats));
}