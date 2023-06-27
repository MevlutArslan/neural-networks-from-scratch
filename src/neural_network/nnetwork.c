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
    * Pointer to the prev layer (Layers can classified as linked lists)

    
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

        layer->prev = prevLayer;
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

        // this is not general
        outputs->elements[i] = network->end->outputs->elements[0];
    }


    network->data->outputs = outputs;
}

// I hope this works :(
void calculateOutputLayerGradient(Layer* outputLayer, Vector* target) {
    for(int i = 0; i < outputLayer->numNeurons; i++) {
        double output = outputLayer->outputs->elements[i];
        double targetValue = target->elements[i];

        // derivative of MSE loss with respect to output
        double dLoss_dOutput = 2 * (output - targetValue);

        // derivative of ReLU with respect to its input
        double dOutput_dInput = output > 0 ? 1 : 0;

        // gradient = dLoss/dOutput * dOutput/dInput
        double gradient = dLoss_dOutput * dOutput_dInput;

        // calculate gradient for each weight
        for(int j = 0; j < outputLayer->neurons[i]->weights->size; j++) {
            // dy/dw = input value for this weight
            double dy_dw = outputLayer->prev->outputs->elements[j];
            outputLayer->neurons[i]->gradient->elements[j] = gradient * dy_dw;
        }

        double dy_dw = 1;  // input for bias is always 1
        outputLayer->neurons[i]->gradient->elements[outputLayer->neurons[i]->weights->size] = gradient * dy_dw;
    }
}

void calculateHiddenLayerGradient(Layer* layer) {
    for(int i = 0; i < layer->numNeurons; i++) {
        // Get output of this neuron
        double output = layer->outputs->elements[i];

        // Derivative of ReLU wrt its input
        double dOutput_dInput = output > 0 ? 1 : 0;

        // The gradient will be the sum of the gradients of all neurons in the next layer, 
        // weighted by the weights connecting this neuron to them.
        double gradient = 0;
        for(int j = 0; j < layer->next->numNeurons; j++) {
            // Get gradient of neuron in next layer. Use j instead of i.
            double nextLayerGradient = layer->next->neurons[j]->gradient->elements[i];

            // Get weight from this neuron to neuron in next layer
            double weight = layer->neurons[i]->weights->elements[j];

            // Add to total gradient
            gradient += nextLayerGradient * weight;
        }

        // Multiply by derivative of ReLU
        gradient *= dOutput_dInput;

        // Assign gradient to all weights of this neuron
        for(int j = 0; j < layer->neurons[i]->weights->size; j++) {
            double dy_dw = layer->prev->outputs->elements[j];  // dy/dw = input value for this weight
            layer->neurons[i]->gradient->elements[j] = gradient * dy_dw;
        }

        // Calculate gradient for the bias
        double dy_dw = 1;  // dy/dw for bias is always 1
        layer->neurons[i]->gradient->elements[layer->neurons[i]->weights->size] = gradient * dy_dw;
    }
}

void updateWeightsAndBiases(NNetwork* network, double learningRate) {
    // Traverse each layer in the network
    Layer* currentLayer = network->start;
    while (currentLayer != NULL) {
        // Traverse each neuron in the current layer
        for (int i = 0; i < currentLayer->numNeurons; i++) {
            printf("GRADIENT:");
            printVector(currentLayer->neurons[i]->gradient);
            printf("BEFORE :");
            printVector(currentLayer->neurons[i]->weights);
            // Traverse each weight in the current neuron
            for (int j = 0; j < currentLayer->neurons[i]->weights->size; j++) {
                // Subtract learning rate times the gradient from the weight
                currentLayer->neurons[i]->weights->elements[j] -= learningRate * currentLayer->neurons[i]->gradient->elements[j];
            }
            // Also update the bias using the last element in the gradient vector
            currentLayer->neurons[i]->bias -= learningRate * currentLayer->neurons[i]->gradient->elements[currentLayer->neurons[i]->weights->size];

            printf("AFTER :");
            printVector(currentLayer->neurons[i]->weights);
        }

        currentLayer = currentLayer->next;
    }
}


void backpropagation(NNetwork* network) {
    printf("LOSS : %f \n", network->loss_function(network->end->outputs, network->data->yValues));
    calculateOutputLayerGradient(network->end, network->data->yValues);

    // skip the output layer as we have already calculated the gradients for it.
    Layer* current = network->end->prev;
    while(current != NULL && current->prev != NULL) {
        calculateHiddenLayerGradient(current);
        current = current->prev;
    }
}