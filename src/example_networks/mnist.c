#include "mnist.h"

// Data
Matrix* trainingData;
Matrix* validationData;
Matrix* yValues_Training;
Matrix* yValues_Testing;

// Plots
gnuplot_ctrl* loss_step_plot;
gnuplot_ctrl* learningRate_step_plot;
gnuplot_ctrl* accuracy_step_plot;

OptimizationConfig create_optimizer(int optimizer) {
    OptimizationConfig optimizationConfig;
    optimizationConfig.optimizer = optimizer;

    // Learning Rate Decay
    optimizationConfig.shouldUseLearningRateDecay = 1;
    
    // optimizationConfig.rho = 0.9;
    optimizationConfig.epsilon = 1e-8;
    optimizationConfig.beta1 = 0.9;
    optimizationConfig.beta2 = 0.999;

    return optimizationConfig;
}


NNetwork* get_network() {
    if(preprocess_data() == -1) {
        log_error("Failed to complete preprocessing of MNIST data!");
    }

    ActivationFunction reluFunc;
    reluFunc.activation = leakyRelu;
    reluFunc.derivative = leakyRelu_derivative;

    NetworkConfig config;
    config.numLayers = 2;
    config.neuronsPerLayer = malloc(sizeof(int) * config.numLayers);
    config.neuronsPerLayer[0] = 128;
    config.neuronsPerLayer[1] = 10;

    config.inputSize = trainingData->columns;

    OptimizationConfig optimizationConfig = create_optimizer(ADAM);

    // if you want to use l1 and/or l2 regularization you need to set the size to config.numLayers and 
    // fill these vectors with the lambda values you want
    config.weightLambdas = create_vector(0);
    config.biasLambdas = create_vector(0);

    if(config.weightLambdas->size > 0 ){
        fill_vector(config.weightLambdas, 1e-5);
    }

    if(config.biasLambdas->size > 0 ){
        fill_vector(config.biasLambdas, 1e-3);
    }

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

    NNetwork* network = create_network(&config);
    network->output = create_matrix(trainingData->rows, network->layers[network->layerCount - 1]->neuronCount);

    return network;
}

int preprocess_data() {
    Data* training_data = load_csv("../archive/mnist_train.csv");
    Data* validation_data = load_csv("../archive/mnist_test.csv");

    if(training_data == NULL) {
        log_error("Failed to load training_data");
        return -1;
    }

    
    if(validation_data == NULL) {
        log_error("Failed to load validation_data");
        return -1;
    }

    // extract training data
    int targetColumn = 0;
    int trainingDataSize = training_data->rows;

    trainingData = get_sub_matrix_except_column(training_data->data, 0, trainingDataSize - 1, 0, training_data->columns - 1, 0);
    
    // extract validation data
    validationData = get_sub_matrix_except_column(validation_data->data, 0, validation_data->rows - 1, 0, validation_data->columns - 1, 0);
    // log_info("Validation Data Matrix: %s", matrix_to_string(validationData));

    // extract yValues
    yValues_Training = oneHotEncode(extractYValues(training_data->data, 0), 10);
    // yValues_Testing = oneHotEncode(extractYValues(validation_data->data, 0), 10);

    // normalize training data 
    for(int col = 0; col < trainingData->columns; col++) {
        normalizeColumn_division(trainingData, col, 255);
    }

    // normalize validation data
    for(int col = 0; col < validationData->columns; col++) {
        normalizeColumn_division(validationData, col, 255);
    }
    
    free_data(training_data);
    free_data(validation_data);

    return 1;
}

void train_network() {
    NNetwork* network = get_network();
    if(network == NULL) {
        log_error("Error creating network!");
        return;
    }

    // default rate of keras -> 0.001
    // kaparthy's recommendation for adam: 0.0003
    double learningRate = 0.01;
    double currentLearningRate = learningRate;
    int step = 1;
    int totalEpochs = 100;

    network->optimizationConfig->learningRateDecayAmount = learningRate / totalEpochs;

    // ------------------------ FOR PLOTTING ------------------------
    double* losses = malloc(sizeof(double) * totalEpochs);
    double* storedSteps = malloc(sizeof(double) * totalEpochs);
    double* learningRates = malloc(sizeof(double) * totalEpochs);
    double* accuracies = malloc(sizeof(double) * totalEpochs);
    // --------------------------------------------------------------
    plot_config();
    double minLoss = __DBL_MAX__;
    double maxAccuracy = 0.0;

      log_info("Starting training with learning rate of: %f for %d epochs.", learningRate, totalEpochs);
    while(step < totalEpochs) {
        learningRates[step] = currentLearningRate;
        
        for(int inputRow = 0; inputRow < trainingData->rows; inputRow++) {
            Vector* output = create_vector(network->layers[network->layerCount - 1]->neuronCount);
            forward_pass(network, trainingData->data[inputRow], output); 
            backpropagation(network, trainingData->data[inputRow], output, yValues_Training->data[inputRow]);
            network->output->data[inputRow] = copy_vector(output);
            free_vector(output);
        }
        
        calculate_loss(network, yValues_Training);
        
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

    log_info("Minimum loss during training: %f \n", minLoss);
    log_info("Maximum accuracy during training: %f \n", maxAccuracy);

    plot_data(losses, storedSteps, learningRates, accuracies, totalEpochs);
}

void plot_data(double* losses, double* storedSteps, double* learningRates, double* accuracies, double totalEpochs) {
    gnuplot_plot_xy(loss_step_plot, storedSteps, losses, totalEpochs, "Loss/Step");
    gnuplot_plot_xy(accuracy_step_plot, storedSteps, accuracies, totalEpochs, "Accuracy/Step");
    gnuplot_plot_xy(learningRate_step_plot, storedSteps, learningRates, totalEpochs, "Learning Rate/Step");


    printf("Press enter to close the plots...\n");
    getchar();
}

void free_plots(double* losses, double* storedSteps, double* learningRates, double* accuracies) {
    free(losses);
    free(storedSteps);
    free(learningRates);
    free(accuracies);

    gnuplot_close(loss_step_plot);
    gnuplot_close(accuracy_step_plot);
    gnuplot_close(learningRate_step_plot);
}

void plot_config() { 
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
}

void validate_network() {

}