#include <stdlib.h>
#include <stdio.h>
#include "neural_network/layer.h"
#include "neural_network/nnetwork.h"
#include "nmath/nmath.h"
#include "helper/matrix_linked_list.h"
#include <string.h>
#include "../tests/test.h"
#include "neural_network/neuron.h"

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

    ActivationFunction reluFunc;
    reluFunc.activation = relu;

    // Create the input vector
    Vector* inputs = createVector(4);
    inputs->elements[0] = 1;
    inputs->elements[1] = 2;
    inputs->elements[2] = 3;
    inputs->elements[3] = 4;

    NetworkConfig config;
    config.numLayers = 3;
    config.neuronsPerLayer = malloc(sizeof(int*) * config.numLayers);
    config.neuronsPerLayer[0] = 3;
    config.neuronsPerLayer[1] = 4;
    config.neuronsPerLayer[2] = 2;

    config.activationFunctions = malloc(sizeof(ActivationFunction) * config.numLayers - 1);  // Allocate memory

    for (int i = 0; i < config.numLayers - 1; i++) {
        config.activationFunctions[i].activation = malloc(sizeof(ActivationFunction));
        config.activationFunctions[i].activation = reluFunc.activation;
    }

    // activation function for output layer
    config.outputActivationFunction = malloc(sizeof(OutputActivationFunction));
    config.outputActivationFunction->activation = softmax;

    NNetwork* network = createNetwork(&config, inputs);

    // Perform forward pass for the network
    forwardPass(network);

    // Retrieve the output vector from the output layer
    Vector* output = network->end->outputs;

    // Print the output values
    printf("Output: ");
    for (int i = 0; i < output->size; i++) {
        printf("%f ", output->elements[i]);
    }
    printf("\n");

    // Clean up memory
    deleteNNetwork(network);
}   