#include <stdlib.h>
#include <stdio.h>
#include "nmath/nmath.h"
#include "neural_network/layer.h"
#include "neural_network/nnetwork.h"
#include <string.h>
#include "../tests/test.h"
#include "helper/data_processing.h"
#include <time.h>
#include "gnuplot_i/gnuplot_i.h"

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

NNetwork* createCustomNetwork() {
    Data* data = loadCSV("/Users/mevlutarslan/Downloads/datasets/Real estate.csv", 0.9f);

    ActivationFunction reluFunc;
    reluFunc.activation = leakyRelu;
    reluFunc.derivative = leakyRelu_derivative;

    LossFunction meanSquaredErrorFunc;
    meanSquaredErrorFunc.loss_function = meanSquaredError;
    meanSquaredErrorFunc.derivative = meanSquaredErrorDerivative;

    NetworkConfig config;
    config.numLayers = 3;
    config.neuronsPerLayer = malloc(sizeof(int) * config.numLayers);
    config.neuronsPerLayer[0] = 3;
    config.neuronsPerLayer[1] = 4;
    config.neuronsPerLayer[2] = 1;

    config.shouldUseGradientClipping = 0;
    config.gradientClippingLowerBound = -1;
    config.gradientClippingUpperBound = 1;

    config.activationFunctions = malloc(sizeof(ActivationFunction) * config.numLayers - 1);  // Allocate memory

    for (int i = 0; i < config.numLayers; i++) {
        config.activationFunctions[i].activation = reluFunc.activation;
        config.activationFunctions[i].derivative = reluFunc.derivative;
    }
    config.data = data;

    config.lossFunction = malloc(sizeof(LossFunction));
    config.lossFunction->loss_function = meanSquaredErrorFunc.loss_function;
    config.lossFunction->derivative = meanSquaredErrorFunc.derivative;


    NNetwork* network = createNetwork(&config);
    

    return network;
}
    /* 
    * input goes into a 3 layer neural network with 3, 4 and 2 neurons.
    * we assign an activation function to each layer (relu in this case)
    * we create a network with this config
    * we run a forward pass
    */
void runProgram() {
    NNetwork* network = createCustomNetwork();
    gnuplot_ctrl* training_plot;
    
    training_plot = gnuplot_init();

    gnuplot_setstyle(training_plot, "points");
    gnuplot_set_xlabel(training_plot, "Steps");
    gnuplot_set_ylabel(training_plot, "Loss");
    double learningRate = 0.0001;
    int steps = 0;
    int maxSteps = 100;
    double* losses = malloc(sizeof(double) * maxSteps);
    double* storedSteps = malloc(sizeof(double) * maxSteps);
    

    while(steps < maxSteps) {
        forwardPass(network);        
        backpropagation(network);
        updateWeightsAndBiases(network, learningRate);
        
        // every x steps
        if(steps % 10 == 0) {
            printf("Step: %d, Loss: %f \n", steps, network->loss);
        }

        losses[steps] = network->loss;
        storedSteps[steps] = steps;

        steps++;
    }

    // Plot loss/step
    gnuplot_plot_xy(training_plot, storedSteps, losses, maxSteps, "Loss/Step");

    printf("Press enter to close plot...\n");
    getchar();

    gnuplot_close(training_plot);
    free(losses);
    free(storedSteps);

    // deleteNNetwork(network);
}   

