#include <stdlib.h>
#include <stdio.h>
#include "nmath/nmath.h"
#include "neural_network/layer.h"
#include "neural_network/nnetwork.h"
#include <string.h>
#include "../tests/test.h"
#include "helper/data_processing.h"
#include <time.h>
#include "../libraries/gnuplot_i/gnuplot_i.h"
#include "../libraries/logger/log.h"

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
        log_info("Running tests...");
        run_tests();
    } else {
        log_info("Running Program!");
        runProgram();
    }
}

NNetwork* createCustomNetwork() {
    Data* data = loadCSV("/Users/mevlutarslan/Downloads/datasets/wine_with_headers.csv", 1, 1);

    ActivationFunction reluFunc;
    reluFunc.activation = leakyRelu;
    reluFunc.derivative = leakyRelu_derivative;

    NetworkConfig config;
    config.numLayers = 2;
    config.neuronsPerLayer = malloc(sizeof(int) * config.numLayers);
    config.neuronsPerLayer[0] = 8;
    config.neuronsPerLayer[1] = 3;

    OptimizationConfig optimizationConfig;
    optimizationConfig.optimizer = ADAM;

     // Learning Rate Decay
    optimizationConfig.shouldUseLearningRateDecay = 0;
    optimizationConfig.learningRateDecayAmount = 1e-8;
    
    // Gradient Clipping
    optimizationConfig.shouldUseGradientClipping = 1;
    optimizationConfig.gradientClippingLowerBound = -1.0;
    optimizationConfig.gradientClippingUpperBound = 1.0;
    
    // Momentum
    optimizationConfig.shouldUseMomentum = 1;
    optimizationConfig.momentum = 0.9;
    
    optimizationConfig.rho = 0.9;
    optimizationConfig.epsilon = 1e-8;
    optimizationConfig.beta1 = 0.9;
    optimizationConfig.beta2 = 0.999;

    config.activationFunctions = malloc(sizeof(ActivationFunction) * config.numLayers - 1);  // Allocate memory
    
    config.optimizationConfig = malloc(sizeof(OptimizationConfig));
    memcpy(config.optimizationConfig, &optimizationConfig, sizeof(OptimizationConfig));
    int i;
    for (i = 0; i < config.numLayers - 1; i++) {
        config.activationFunctions[i].activation = reluFunc.activation;
        config.activationFunctions[i].derivative = reluFunc.derivative;
    }

    config.activationFunctions[config.numLayers - 1].activation = softmax;
    config.activationFunctions[config.numLayers - 1].derivative = softmax_derivative;

    config.data = data;

    config.lossFunction = malloc(sizeof(LossFunction));
    config.lossFunction->loss_function = crossEntropyLoss;
    config.lossFunction->derivative = crossEntropyLossDerivative;

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

    double learningRate = 0.0001;
    double currentLearningRate = learningRate;
    int step = 0;
    int maxSteps = 1;

    // ------------------------ FOR PLOTTING ------------------------
    double* losses = malloc(sizeof(double) * maxSteps);
    double* storedSteps = malloc(sizeof(double) * maxSteps);
    double* learningRates = malloc(sizeof(double) * maxSteps);
    // --------------------------------------------------------------

    double minLoss = __DBL_MAX__;
    
    while(step < 5) {
        learningRates[step] = currentLearningRate;
        // forwardPass(network, network->data->trainingData, network->data->trainingOutputs);
        // backpropagation(network);
        
        // if(network->optimizationConfig->shouldUseLearningRateDecay == 1) {
        //     double decayRate = network->optimizationConfig->learningRateDecayAmount;
        //     currentLearningRate = learningRate * (1 / (1.0 + (decayRate * (double)step)));
        // }

        // for(int i = 0; i < network->layerCount; i++) {
        //     // printMatrix(network->layers[i]->gradients);
        // }
        // network->currentStep = step;
        // network->optimizer(network, currentLearningRate);

        // if(step % 100 == 0){
        //     if(network->lossFunction->loss_function == crossEntropyLoss) {
        //         double acc = accuracy(network->data->yValues, network->data->trainingOutputs);
        //         printf("Step: %d, Accuracy: %f, Loss: %f \n", step, acc, network->loss);  
        //     }
        // }
        // // else{
        // //     printf("Step: %d, Loss: %f \n", step, network->loss);  
        // // }
        // minLoss = fmin(minLoss, network->loss);

        // losses[step] = network->loss;
        // storedSteps[step] = step;
        step++;
    }

    // double acc = accuracy(network->data->yValues, network->data->trainingOutputs);
    // printf("Step: %d, Accuracy: %f, Loss: %f \n", step, acc, network->loss);  

    // printf("MIN LOSS TRAINING: %f \n", minLoss);

    // Plot loss/step
    gnuplot_plot_xy(loss_step_plot, storedSteps, losses, maxSteps, "Loss/Step");
    // gnuplot_plot_xy(learningRate_step_plot, storedSteps, learningRates, maxSteps, "Learning Rate/Step");

    // Matrix* evaluationOutputs = createMatrix(network->data->evaluationData->rows, network->layers[network->layerCount - 1]->neuronCount);
    
    // forwardPass(network, network->data->evaluationData, evaluationOutputs);

    // // unnormalize the output
    // // i have the same number of rows as the evaluation data, and I have 1 column which i need to normalize
    // for(int i = 0; i < evaluationOutputs->rows; i++) {
    //     unnormalizeVector(evaluationOutputs->data[i], network->data->minValues->elements[network->data->minValues->size - 1], network->data->maxValues->elements[network->data->maxValues->size - 1]);
    // }

    // // unnormalize the yValues
    // Vector* unnormalizedYValues = spliceVector(network->data->yValues, network->data->trainingData->rows + 1, network->data->numberOfRows);
    // unnormalizeVector(unnormalizedYValues, network->data->minValues->elements[network->data->minValues->size - 1], network->data->maxValues->elements[network->data->maxValues->size - 1]);
    
    // // printMatrix(network->layers[1]->weights);

    printf("Press enter to close plot...\n");
    getchar();

    gnuplot_close(loss_step_plot);
    // gnuplot_close(learningRate_step_plot);

    free(losses);
    free(storedSteps);

    // freeMatrix(evaluationOutputs);
    // freeVector(unnormalizedYValues);

    // deleteNNetwork(network);
}   

