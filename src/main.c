#include <stdlib.h>
#include <stdio.h>
#include "nmath/nmath.h"
#include "neural_network/layer.h"
#include "neural_network/nnetwork.h"
#include <string.h>
#include "../tests/test.h"
#include "helper/data_processing.h"
#include <time.h>

void runProgram();

int main(int argc, char* argv[])
{
    srand(time(NULL));

    int isTesting = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "test") == 0) {
            isTesting = 1;
            break;
        }
    }

    if (isTesting) {
        printf("Running tests...\n");
        run_tests();
    } else {
        printf("Running the program...\n");
        runProgram();
    }
}

    /* 
    * input goes into a 3 layer neural network with 3, 4 and 2 neurons.
    * we assign an activation function to each layer (relu in this case)
    * we create a network with this config
    * we run a forward pass
    */
void runProgram() {
    Data* data = loadCSV("/Users/mevlutarslan/Downloads/datasets/Real estate.csv", 0.9f);

    ActivationFunction reluFunc;
    reluFunc.activation = leakyRelu;
    // reluFunc.derivative = relu_derivative;

    LossFunction meanSquaredErrorFunc;
    meanSquaredErrorFunc.loss_function = meanSquaredError;

    // Create the input vector
    Matrix* input = data->trainingData;

    NetworkConfig config;
    config.numLayers = 3;
    config.neuronsPerLayer = malloc(sizeof(int) * config.numLayers);
    config.neuronsPerLayer[0] = 3;
    config.neuronsPerLayer[1] = 4;
    config.neuronsPerLayer[2] = 1;

    config.activationFunctions = malloc(sizeof(ActivationFunction) * config.numLayers - 1);  // Allocate memory

    for (int i = 0; i < config.numLayers; i++) {
        config.activationFunctions[i].activation = reluFunc.activation;
    }

    config.lossFunction = &meanSquaredError;

    config.data = data;

    NNetwork* network = createNetwork(&config);
    
    network->lossFunction = &meanSquaredErrorFunc;

    double learningRate = 0.00001;
    int steps = 0;

    while(steps < 1000) {
        forwardPass(network);        
        backpropagation(network);
        updateWeightsAndBiases(network, learningRate);
        
        steps++;
    }
   
    // Clean up memory
    // deleteNNetwork(network);
}   