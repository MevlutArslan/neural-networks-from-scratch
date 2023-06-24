#include <stdlib.h>
#include <stdio.h>
#include "neural_network/layer.h"
#include "neural_network/nnetwork.h"
#include "nmath/nmath.h"
#include "helper/matrix_linked_list.h"
#include <string.h>
#include "../tests/test.h"
#include "neural_network/neuron.h"
#include "helper/data_processing/data_processing.h"

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
    * vector of size 4 goes into a 3 layer neural network with 3, 4 and 2 neurons.
    * we assign an activation function to each layer (relu in this case)
    * we create a network with this config
    * we run a forward pass
    */
void runProgram() {

    Data* data = loadCSV("/Users/mevlutarslan/Downloads/datasets/Real estate.csv", 0.9f);

    ActivationFunction reluFunc;
    reluFunc.activation = relu;

    // Create the input vector
    Matrix* input = data->trainingData;

    NetworkConfig config;
    config.numLayers = 3;
    config.neuronsPerLayer = malloc(sizeof(int*) * config.numLayers);
    config.neuronsPerLayer[0] = 3;
    config.neuronsPerLayer[1] = 4;
    config.neuronsPerLayer[2] = 1;

    config.activationFunctions = malloc(sizeof(ActivationFunction) * config.numLayers - 1);  // Allocate memory

    for (int i = 0; i < config.numLayers - 1; i++) {
        config.activationFunctions[i].activation = malloc(sizeof(ActivationFunction));
        config.activationFunctions[i].activation = reluFunc.activation;
    }

    // activation function for output layer
    config.outputActivationFunction = malloc(sizeof(OutputActivationFunction));
    config.outputActivationFunction->activation = applyReLU;

    NNetwork* network = createNetwork(&config, input->data[0]);

    // Perform forward pass for the network
    forwardPass(network, input);
    
    // Clean up memory
    deleteNNetwork(network);
}   