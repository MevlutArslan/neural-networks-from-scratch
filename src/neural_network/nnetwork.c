#include "nnetwork.h"


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
    * Pointer to the prev layer (Layers can classified as linked lists)

    
    we loop config.number of layers times and create each layer.
    we mark the first layer to be network's start layer and the last layer to be the end layer
*/
NNetwork* createNetwork(const NetworkConfig* config) {
    NNetwork* network = malloc(sizeof(NNetwork));
    network->data = config->data;
    network->layerCount = config->numLayers;
    network->layers = malloc(network->layerCount * sizeof(Layer));

   for (int i = 0; i < config->numLayers; i++) {
        LayerConfig layerConfig;
        layerConfig.inputSize = i == 0 ? config->data->numberOfColumns : network->layers[i - 1]->output->size;
        layerConfig.neuronCount = config->neuronsPerLayer[i];
        layerConfig.activationFunction = &config->activationFunctions[i];
        
        Layer* layer = createLayer(&layerConfig);

        layer->next = i < config->numLayers - 1 ? network->layers[i + 1] : NULL;
        layer->prev = i > 0 ? network->layers[i - 1] : NULL;

        network->layers[i] = layer;
    }


    network->loss_function = config->loss_function;
    network->loss = createVector(network->data->trainingData->rows);

    return network;
}

void forwardPass(NNetwork* network) {
    network->data->trainingOutputs = createMatrix(network->data->trainingData->rows, network->layers[network->layerCount - 1]->neuronCount);
    for (int i = 0; i < network->data->trainingData->rows; i++) {
        network->layers[0]->input = reshapeVectorToMatrix(network->data->trainingData->data[i]);
        for (int layerIndex = 0; layerIndex < network->layerCount; layerIndex++) {
            Layer* currentLayer = network->layers[layerIndex];
            Matrix* weightedSum = matrix_dot_product(matrix_transpose(currentLayer->input), currentLayer->weights);
            
            Vector* outputVector = matrixToVector(weightedSum);
            
            Vector* outputWithBias = vector_addition(outputVector, currentLayer->biases);

            currentLayer->output = outputWithBias;

            currentLayer->activationFunction->activation(currentLayer->output);

            if(layerIndex != network->layerCount - 1) 
                network->layers[layerIndex + 1]->input = reshapeVectorToMatrix(currentLayer->output);

            freeMatrix(weightedSum);
            freeVector(outputVector);
        }
        network->data->trainingOutputs->data[i] = copyVector(network->layers[network->layerCount - 1]->output);
    }
}

void deleteNNetwork(NNetwork* network){
    deleteLayer(network->layers[0]);
}
