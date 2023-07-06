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
        network->layers[i] = layer;
    }

    network->lossFunction = config->lossFunction;

    return network;
}

void forwardPass(NNetwork* network) {
    network->data->trainingOutputs = createMatrix(network->data->trainingData->rows, network->layers[network->layerCount - 1]->neuronCount);
    
    for (int i = 0; i < network->data->trainingData->rows; i++) {
        network->layers[0]->input = network->data->trainingData->data[i];

        for (int layerIndex = 0; layerIndex < network->layerCount; layerIndex++) {
            Layer* currentLayer = network->layers[layerIndex];
            Vector* weightedSum = dot_product(currentLayer->weights, currentLayer->input);

            currentLayer->weightedSums = copyVector(weightedSum);

            currentLayer->output = vector_addition(weightedSum, currentLayer->biases);;

            currentLayer->activationFunction->activation(currentLayer->output);
            
            if(layerIndex != network->layerCount - 1) {
                network->layers[layerIndex + 1]->input = currentLayer->output;
            }
        }
        network->data->trainingOutputs->data[i] = copyVector(network->layers[network->layerCount - 1]->output);
    }

    network->loss = meanSquaredError(network->data->trainingOutputs, network->data->yValues);
}

void backpropagation(NNetwork* network) {

    Layer* outputLayer = network->layers[network->layerCount - 1];

    // for each output
    for(int outputIndex = 0; outputIndex < network->data->trainingData->rows; outputIndex++) {
        
        // the output layer's step
        int layerIndex = network->layerCount - 1;
        Layer* currentLayer = network->layers[layerIndex];
        for(int outputNeuronIndex = 0; outputNeuronIndex < outputLayer->neuronCount; outputNeuronIndex++) {
            Vector* predictions = network->data->trainingOutputs->data[outputIndex];
            double prediction = predictions->elements[outputNeuronIndex];

            double target = network->data->yValues->elements[outputIndex];
            double error =  target - prediction;
            error *= error;
            error *= 0.5;
            /* TODO: ABSTRACT THIS TO SUIT MULTIPLE LOSS FUNCTIONS  
               derivative of MSE is -1 * error, as:
               derivative of 1/2 * (value)^2 = 1/2 * 2(value) => 2/2 * (value) = 1 * value
            
               Using the chain rule for differentiation (f(g(x)) = df(g(x)) * g'(x)), we then multiply this by the derivative of the inner function,
               g(x) = (desired - predicted), with respect to 'predicted', which gives g'(x) = -1
               Therefore, the derivative of the MSE with respect to 'predicted' is: f'(g(predicted)) * g'(predicted) = (desired - predicted) * -1 = predicted - desired
            */
           
            double dLoss_dOutput = network->lossFunction->derivative(target, prediction);

            double dOutput_dWeightedSum = currentLayer->activationFunction->derivative(currentLayer->weightedSums->elements[outputNeuronIndex]);
            double dLoss_dWeightedSum = dLoss_dOutput * dOutput_dWeightedSum;

            for(int weightIndex = 0; weightIndex < currentLayer->weights->columns; weightIndex++) {
                double dWeightedSum_dWeight = 0.0f;
                if(layerIndex == 0) {
                    dWeightedSum_dWeight= network->layers[layerIndex]->input->elements[weightIndex];
                }else {
                    dWeightedSum_dWeight = network->layers[layerIndex-1]->output->elements[weightIndex];
                }

                double dLoss_dWeight = dLoss_dWeightedSum * dWeightedSum_dWeight;
                
                if(network->shouldUseGradientClipping == 1) {
                    if(dLoss_dWeight < network->gradientClippingLowerBound) {
                        dLoss_dWeight = network->gradientClippingLowerBound;
                    }else if(dLoss_dWeight > network->gradientClippingUpperBound) {
                        dLoss_dWeight = network->gradientClippingUpperBound;
                    }
                }

                currentLayer->gradients->data[outputNeuronIndex]->elements[weightIndex] += dLoss_dWeight;                
            }
            
            currentLayer->biasGradients->elements[outputNeuronIndex] = dLoss_dOutput;
            currentLayer->dLoss_dWeightedSums->elements[outputNeuronIndex] = dLoss_dOutput * dOutput_dWeightedSum;
        }

        for(layerIndex = network->layerCount - 2; layerIndex >= 0; layerIndex --) {
            currentLayer = network->layers[layerIndex];
            for(int neuronIndex = 0; neuronIndex < currentLayer->neuronCount; neuronIndex++) {
                double dLoss_dOutput = 0.0f;
                
                Layer* nextLayer = network->layers[layerIndex + 1];
                for(int neuronIndexNext = 0; neuronIndexNext < nextLayer->neuronCount; neuronIndexNext++) {
                    dLoss_dOutput += nextLayer->dLoss_dWeightedSums->elements[neuronIndexNext] * nextLayer->weights->data[neuronIndexNext]->elements[neuronIndex];
                }
                double dOutput_dWeightedSum = currentLayer->activationFunction->derivative(currentLayer->weightedSums->elements[neuronIndex]);

                double dLoss_dWeightedSum = dLoss_dOutput * dOutput_dWeightedSum;

                for(int weightIndex = 0; weightIndex < currentLayer->weights->columns; weightIndex++) {
                    double dWeightedSum_dWeight = 0.0f;
                    if(layerIndex == 0) {
                        dWeightedSum_dWeight= network->layers[layerIndex]->input->elements[weightIndex];
                    }else {
                        dWeightedSum_dWeight = network->layers[layerIndex-1]->output->elements[weightIndex];
                    }
                    
                    double dLoss_dWeight = dLoss_dWeightedSum * dWeightedSum_dWeight;
                    if(network->shouldUseGradientClipping == 1) {
                        if(dLoss_dWeight < network->gradientClippingLowerBound) {
                            dLoss_dWeight = network->gradientClippingLowerBound;
                        }else if(dLoss_dWeight > network->gradientClippingUpperBound) {
                            dLoss_dWeight = network->gradientClippingUpperBound;
                        }
                    }
                    currentLayer->biasGradients->elements[neuronIndex] = dLoss_dOutput;
                    currentLayer->gradients->data[neuronIndex]->elements[weightIndex] += dLoss_dWeight;
                }

                currentLayer->dLoss_dWeightedSums->elements[neuronIndex] = dLoss_dOutput * dOutput_dWeightedSum;
            }
        }
    }
}

void updateWeightsAndBiases(NNetwork* network, double learningRate) {
    for(int layerIndex = 0; layerIndex < network->layerCount; layerIndex++) {
        Layer* currentLayer = network->layers[layerIndex];

        for(int neuronIndex = 0; neuronIndex < currentLayer->neuronCount; neuronIndex++) {
            for(int weightIndex = 0; weightIndex < currentLayer->weights->columns; weightIndex++) {
                double gradient = currentLayer->gradients->data[neuronIndex]->elements[weightIndex];
                
                currentLayer->weights->data[neuronIndex]->elements[weightIndex] -= learningRate * gradient;
            }
            currentLayer->biases->elements[neuronIndex] -= learningRate * currentLayer->biasGradients->elements[neuronIndex];
        }
    }
}

void deleteNNetwork(NNetwork* network){
    for(int i = network->layerCount - 1; i >= 0; i--) {
        deleteLayer(network->layers[i]);
    }
}

void dumpNetworkState(NNetwork* network) {
    printf("------------------------------ Network State ------------------------------\n");
    printf("Loss: %f\n", network->loss);

    // Dump information for each layer
    for (int layerIndex = 0; layerIndex < network->layerCount; layerIndex++) {
        Layer* currentLayer = network->layers[layerIndex];

        printf("------------------------ Layer %d ------------------------\n", layerIndex);
        printf("Neuron count: %d\n", currentLayer->neuronCount);

        // Print input matrix
        printf("Input matrix:\n");
        printMatrix(currentLayer->input);

        // Print weights matrix
        printf("Weights matrix:\n");
        printMatrix(currentLayer->weights);

        // Print biases vector
        printf("Biases vector:\n");
        printVector(currentLayer->biases);

        // Print weighted sums vector
        printf("Weighted sums vector:\n");
        printVector(currentLayer->weightedSums);

        // Print output vector
        printf("Output vector:\n");
        printVector(currentLayer->output);

        // // Print error vector
        // printf("Error vector:\n");
        // printVector(currentLayer->error);

        // Print gradients matrix
        printf("Gradients matrix:\n");
        printMatrix(currentLayer->gradients);

        // Print other layer information here
        printf("------------------------------------------------------------\n");
    }

    printf("------------------------------------------------------------------------\n");
}
