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
    //for testing only, otherwise set to time(NULL)
    srand(306);

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
    config.numLayers = 2;
    config.neuronsPerLayer = malloc(sizeof(int) * config.numLayers);
    config.neuronsPerLayer[0] = 3;
    config.neuronsPerLayer[1] = 1;
    // config.neuronsPerLayer[2] = 1;

    OptimizationConfig optimizationConfig;
    // Learning Rate Decay
    optimizationConfig.shouldUseLearningRateDecay = 1;
    optimizationConfig.learningRateDecayAmount = 1e-2;
    
    // Gradient Clipping
    optimizationConfig.shouldUseGradientClipping = 1;
    optimizationConfig.gradientClippingLowerBound = -0.5;
    optimizationConfig.gradientClippingUpperBound = 0.5;
    
    // Momentum
    optimizationConfig.shouldUseMomentum = 1;
    optimizationConfig.momentum = 0.3;

    optimizationConfig.optimizer = RMS_PROP;
    optimizationConfig.epsilon = 1e-7;
    optimizationConfig.rho = 0.9;

    config.activationFunctions = malloc(sizeof(ActivationFunction) * config.numLayers - 1);  // Allocate memory
    
    config.optimizationConfig = malloc(sizeof(OptimizationConfig));
    memcpy(config.optimizationConfig, &optimizationConfig, sizeof(OptimizationConfig));

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
    gnuplot_ctrl* loss_step_plot;
    gnuplot_ctrl* learningRate_step_plot;

    loss_step_plot = gnuplot_init();
    learningRate_step_plot = gnuplot_init();


    gnuplot_setstyle(loss_step_plot, "linespoints");
    
    gnuplot_set_xlabel(loss_step_plot, "Steps");
    gnuplot_set_ylabel(loss_step_plot, "Loss");

    gnuplot_setstyle(learningRate_step_plot, "linespoints");
    gnuplot_set_xlabel(learningRate_step_plot, "Steps");
    gnuplot_set_ylabel(learningRate_step_plot, "Loss");

    double learningRate = 0.01;
    double currentLearningRate = learningRate;
    int steps = 0;
    int maxSteps = 40;

    // ------------------------ FOR PLOTTING ------------------------
    double* losses = malloc(sizeof(double) * maxSteps);
    double* storedSteps = malloc(sizeof(double) * maxSteps);
    double* learningRates = malloc(sizeof(double) * maxSteps);
    // --------------------------------------------------------------

    double minLoss = __DBL_MAX__;

    while(steps < maxSteps) {
        learningRates[steps] = currentLearningRate;
        forwardPass(network);        
        backpropagation(network);

        if(network->optimizationConfig->shouldUseLearningRateDecay == 1) {
            double decayRate = network->optimizationConfig->learningRateDecayAmount;
            currentLearningRate = learningRate * (1 / (1.0 + (decayRate * (double)steps)));
        }

        network->optimizer(network, currentLearningRate);
        
        printf("Step: %d, Loss: %f \n", steps, network->loss);  

        minLoss = fmin(minLoss, network->loss);

        losses[steps] = network->loss;
        storedSteps[steps] = steps;
        steps++;
    }
    printf("MIN LOSS: %f \n", minLoss);

    // Plot loss/step
    gnuplot_plot_xy(loss_step_plot, storedSteps, losses, maxSteps, "Loss/Step");
    gnuplot_plot_xy(learningRate_step_plot, storedSteps, learningRates, maxSteps, "Learning Rate/Step");

    printf("Press enter to close plot...\n");
    getchar();

    gnuplot_close(loss_step_plot);
    gnuplot_close(learningRate_step_plot);

    free(losses);
    free(storedSteps);

    // deleteNNetwork(network);
}   

