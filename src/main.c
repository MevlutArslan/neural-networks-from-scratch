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

FILE* logFile;
Matrix* trainingData;
Matrix* validationData;
Matrix* yValues;

void file_log(log_Event *ev);
NNetwork* create_custom_network();

OptimizationConfig create_optimizer(int optimizer) {
    OptimizationConfig optimizationConfig;
    optimizationConfig.optimizer = optimizer;

    // Learning Rate Decay
    optimizationConfig.shouldUseLearningRateDecay = 1;
    
    // Gradient Clipping
    optimizationConfig.shouldUseGradientClipping = 0;
    optimizationConfig.gradientClippingLowerBound = -1.0;
    optimizationConfig.gradientClippingUpperBound = 1.0;
    
    // Momentum
    optimizationConfig.shouldUseMomentum = 1;
    optimizationConfig.momentum = 0.9;
    
    // optimizationConfig.rho = 0.9;
    optimizationConfig.epsilon = 1e-8;
    optimizationConfig.beta1 = 0.9;
    optimizationConfig.beta2 = 0.999;

    return optimizationConfig;
}


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


void runProgram() {
    logFile = fopen("log.txt", "w");
    if (logFile == NULL) {
        printf("Failed to open log file.\n");
        return;
    }

    // Add the file_log as a callback to the logging library
    log_add_callback(file_log, NULL, LOG_TRACE);

    // log_info("Debug %s \n", DEBUG == 1? "is enabled" : "is diabled");

    NNetwork* network = create_custom_network();

    // ------------------------ FOR PLOTTING ------------------------
    gnuplot_ctrl* loss_step_plot;
    gnuplot_ctrl* learningRate_step_plot;
    gnuplot_ctrl* accuracy_step_plot;
    
    loss_step_plot = gnuplot_init();
    learningRate_step_plot = gnuplot_init();
    accuracy_step_plot = gnuplot_init();

    gnuplot_setstyle(loss_step_plot, "linespoints");
    gnuplot_set_xlabel(loss_step_plot, "Steps");
    gnuplot_set_ylabel(loss_step_plot, "Loss");

    gnuplot_setstyle(accuracy_step_plot, "linespoints");
    gnuplot_set_xlabel(accuracy_step_plot, "Steps");
    gnuplot_set_ylabel(accuracy_step_plot, "Loss");

    gnuplot_setstyle(learningRate_step_plot, "dots");
    gnuplot_set_xlabel(learningRate_step_plot, "Steps");
    gnuplot_set_ylabel(learningRate_step_plot, "Learning Rate");

    // --------------------------------------------------------------

    // default rate of keras -> 0.001
    // kaparthy's recommendation for adam: 0.0003
    double learningRate = 0.01;
    double currentLearningRate = learningRate;
    int step = 1;
    int maxSteps = 250;

    network->optimizationConfig->learningRateDecayAmount = learningRate / maxSteps;

    // // ------------------------ FOR PLOTTING ------------------------
    double* losses = malloc(sizeof(double) * maxSteps);
    double* storedSteps = malloc(sizeof(double) * maxSteps);
    double* learningRates = malloc(sizeof(double) * maxSteps);
    double* accuracies = malloc(sizeof(double) * maxSteps);
    // // --------------------------------------------------------------

    double minLoss = __DBL_MAX__;
    double maxAccuracy = 0.0;
        
    double momentum = network->optimizationConfig->momentum;
    log_info("Starting training with learning rate of: %f for %d epochs.", learningRate, maxSteps);
    while(step < maxSteps) {
        learningRates[step] = currentLearningRate;
        
        for(int inputRow = 0; inputRow < trainingData->rows; inputRow++) {
            Vector* output = create_vector(network->layers[network->layerCount - 1]->neuronCount);
            forwardPass(network, trainingData->data[inputRow], output); 
            backpropagation(network, trainingData->data[inputRow], output, yValues->data[inputRow]);
            network->output->data[inputRow] = copy_vector(output);
            free_vector(output);
        }
        
        calculateLoss(network, yValues);
        
        if(network->optimizationConfig->shouldUseLearningRateDecay == 1) {
            double decayRate = network->optimizationConfig->learningRateDecayAmount;
            currentLearningRate = currentLearningRate * (1 / (1.0 + (decayRate * (double)step)));
        }
        network->currentStep = step;
        network->optimizer(network, currentLearningRate);

        // if(step == 1 || step % 10 == 0){
            log_info("Step: %d, Accuracy: %f, Loss: %f \n", step, network->accuracy, network->loss);  
        // }
        minLoss = fmin(minLoss, network->loss);
        
        maxAccuracy = fmax(maxAccuracy, network->accuracy);

        losses[step] = network->loss;
        storedSteps[step] = step;
        accuracies[step] = network->accuracy;

        step++;
        // Clear the gradients
        for(int layerIndex = 0; layerIndex < network->layerCount; layerIndex++) {
            fill_matrix(network->layers[layerIndex]->gradients, 0.0f);
            fill_vector(network->layers[layerIndex]->biasGradients, 0.0f);
        }
    }

    log_info("Minimum loss: %f \n", minLoss);
    log_info("Maximum accuracy: %f \n", maxAccuracy);

    // Plot loss/step
    gnuplot_plot_xy(loss_step_plot, storedSteps, losses, maxSteps, "Loss/Step");
    gnuplot_plot_xy(accuracy_step_plot, storedSteps, accuracies, maxSteps, "Accuracy/Step");

    gnuplot_plot_xy(learningRate_step_plot, storedSteps, learningRates, maxSteps, "Learning Rate/Step");

    // Matrix* validationOutputs = create_matrix(validationData->rows, network->layers[network->layerCount - 1]->neuronCount);
    // forwardPass(network, validationData);

    // // // unnormalize the output
    // // // i have the same number of rows as the evaluation data, and I have 1 column which i need to normalize
    // int correctPredictions = 0;
    // for(int i = 0; i < validationOutputs->rows; i++) {
    //     log_info("Target for Index %d: %s \n", i, vector_to_string(yValues->data[trainingData->rows + i]));
    //     log_info("Prediction for Index %d: %s \n", i, vector_to_string(validationOutputs->data[i]));

    //     if(arg_max(yValues->data[trainingData->rows + i]) == arg_max(validationOutputs->data[i])) {
    //         correctPredictions++;
    //     }
    // }

    // log_debug("Accuracy with evaluation data: %f \n", correctPredictions / (double)validationOutputs->rows);
    printf("Press enter to close plot...\n");
    getchar();

    gnuplot_close(loss_step_plot);
    gnuplot_close(accuracy_step_plot);
    gnuplot_close(learningRate_step_plot);

    free(losses);
    free(storedSteps);
    // free_matrix(validationOutputs);

    // deleteNNetwork(network);
    fclose(logFile);
}   

void file_log(log_Event *ev) {
  fprintf(
    logFile, "%s %-5s %s:%d: ",
    ev->time, log_level_string(ev->level), ev->file, ev->line);
  vfprintf(logFile, ev->fmt, ev->ap);
  fprintf(logFile, "\n");
  fflush(logFile);
}

NNetwork* create_custom_network() {
    Data* data = load_csv("/Users/mevlutarslan/Downloads/datasets/wine_with_headers.csv");
    if(data == NULL) {
        log_error("Failed to load CSV");
    }

    // shuffling data to have various classes in both my training and validation sets
    shuffle_rows(data->data);

    // amount to split the dataset by.
    double splitPercentage = 0.8;

    // extract training data
    int targetColumn = 0;
    int trainingDataSize = data->rows * splitPercentage;
    trainingData = get_sub_matrix_except_column(data->data, 0, trainingDataSize - 1, 0, data->columns - 1, 0);
    
    // extract validation data
    validationData = get_sub_matrix_except_column(data->data, trainingData->rows + 1, data->rows - 1, 0, data->columns - 1, 0);
    // log_info("Validation Data Matrix: %s", matrix_to_string(validationData));

    // extract yValues
    yValues = oneHotEncode(extractYValues(data->data, 0), 3);
    // log_info("Y Values Matrix: %s", matrix_to_string(yValues));

    // normalize training data 
    for(int col = 0; col < trainingData->columns; col++) {
        normalizeColumn(trainingData, col);
    }
    log_info("Training Data Matrix: %s", matrix_to_string(trainingData));


    // normalize validation data
    for(int col = 0; col < validationData->columns; col++) {
        normalizeColumn(validationData, col);
    }

    ActivationFunction reluFunc;
    reluFunc.activation = leakyRelu;
    reluFunc.derivative = leakyRelu_derivative;

    NetworkConfig config;
    config.numLayers = 2;
    config.neuronsPerLayer = malloc(sizeof(int) * config.numLayers);
    config.neuronsPerLayer[0] = 2;
    config.neuronsPerLayer[1] = 3;

    config.inputSize = trainingData->columns;

    OptimizationConfig optimizationConfig = create_optimizer(ADAM);

    config.activationFunctions = malloc(sizeof(ActivationFunction) * config.numLayers - 1);  // Allocate memory
    
    config.optimizationConfig = malloc(sizeof(OptimizationConfig));
    memcpy(config.optimizationConfig, &optimizationConfig, sizeof(OptimizationConfig));

    int i;
    for (i = 0; i < config.numLayers - 1; i++) {
        config.activationFunctions[i].activation = reluFunc.activation;
        config.activationFunctions[i].derivative = reluFunc.derivative;
    }

    config.activationFunctions[config.numLayers - 1].activation = softmax;

    config.lossFunction = malloc(sizeof(LossFunction));
    config.lossFunction->loss_function = categoricalCrossEntropyLoss;

    NNetwork* network = createNetwork(&config);
    network->output = create_matrix(trainingData->rows, network->layers[network->layerCount - 1]->neuronCount);

    return network;
}
