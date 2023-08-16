#include "mnist.h"

NNetwork* mnist_get_network(Model* model);
void mnist_train_network(Model* model);
void mnist_validate_network(Model* model);
int mnist_preprocess_data(ModelData* modelData);
void mnist_plot_data(ModelData* modelData);
void mnist_plot_config();

Model* create_mnist_model() {
    
    Model* model = malloc(sizeof(Model));

    model->get_network = &mnist_get_network;
    model->train_network = &mnist_train_network;
    model->validate_network = &mnist_validate_network;
    model->preprocess_data = &mnist_preprocess_data;
    model->plot_data = &mnist_plot_data;
    model->plot_config = &mnist_plot_config;

    model->data = (ModelData*) malloc(sizeof(ModelData));
    model->data->totalEpochs = 250;
    model->data->losses = malloc(sizeof(double) * model->data->totalEpochs);
    model->data->epochs = malloc(sizeof(double) * model->data->totalEpochs);
    model->data->learningRates = malloc(sizeof(double) * model->data->totalEpochs);
    model->data->accuracies = malloc(sizeof(double) * model->data->totalEpochs);
    model->data->path = "mnist_example_network";

    return model;
}

OptimizationConfig create_optimizer(int optimizer) {
    OptimizationConfig optimizationConfig;
    optimizationConfig.optimizer = optimizer;

    // Learning Rate Decay
    optimizationConfig.shouldUseLearningRateDecay = 1;
    optimizationConfig.shouldUseGradientClipping = 0;

    // optimizationConfig.rho = 0.9;
    optimizationConfig.epsilon = 1e-8;
    optimizationConfig.beta1 = 0.9;
    optimizationConfig.beta2 = 0.999;

    return optimizationConfig;
}

NNetwork* mnist_get_network(Model* model) {
    if(model->preprocess_data(model->data) == -1) {
        log_error("%s", "Failed to complete preprocessing of MNIST data!");
    }

    NetworkConfig config;
    config.numLayers = 2;
    config.neuronsPerLayer = malloc(sizeof(int) * config.numLayers);
    config.neuronsPerLayer[0] = 128;
    config.neuronsPerLayer[1] = 10;

    config.inputSize = model->data->trainingData->columns;

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
    for (int i = 0; i < config.numLayers - 1; i++) {
        memcpy(&config.activationFunctions[i], &LEAKY_RELU, sizeof(ActivationFunction));
    }

    memcpy(&config.activationFunctions[config.numLayers - 1], &SOFTMAX, sizeof(ActivationFunction));

    config.lossFunction = malloc(sizeof(LossFunction));
    memcpy(&config.lossFunction->loss_function, &CATEGORICAL_CROSS_ENTROPY, sizeof(LossFunction));


    NNetwork* network = create_network(&config);
    network->output = create_matrix(model->data->trainingData->rows, network->layers[network->layerCount - 1]->neuronCount);

    free_network_config(&config);

    return network;
}

int mnist_preprocess_data(ModelData* modelData) {
    Data* training_data = load_csv("/Users/mevlutarslan/Downloads/datasets/mnistcsv/mnist_train.csv");
    Data* validation_data = load_csv("/Users/mevlutarslan/Downloads/datasets/mnistcsv/mnist_test.csv");

    if(training_data == NULL) {
        log_error("%s", "Failed to load training_data");
        return -1;
    }

    
    if(validation_data == NULL) {
        log_error("%s", "Failed to load validation_data");
        return -1;
    }

    // extract training data
    int targetColumn = 0;
    int trainingDataSize = training_data->rows;

    modelData->trainingData = get_sub_matrix_except_column(training_data->data, 0, trainingDataSize - 1, 0, training_data->columns - 1, 0);
    
    // extract validation data
    modelData->validationData = get_sub_matrix_except_column(validation_data->data, 0, validation_data->rows - 1, 0, validation_data->columns - 1, 0);
    log_debug("Validation Data Matrix: %s", matrix_to_string(modelData->validationData));

    // extract yValues
    Vector* yValues_Training = extractYValues(training_data->data, 0);
    Vector* yValues_Testing = extractYValues(validation_data->data, 0);

    modelData->yValues_Training = oneHotEncode(yValues_Training, 10);
    modelData->yValues_Testing = oneHotEncode(yValues_Testing, 10);

    // normalize training data 
    for(int col = 0; col < modelData->trainingData->columns; col++) {
        normalizeColumn_division(modelData->trainingData, col, 255);
    }

    // normalize validation data
    for(int col = 0; col < modelData->validationData->columns; col++) {
        normalizeColumn_division(modelData->validationData, col, 255);
    }
    
    free_data(training_data);
    free_data(validation_data);
    free_vector(yValues_Training);
    free_vector(yValues_Testing);

    return 1;
}

void mnist_train_network(Model* model) {
    NNetwork* network = mnist_get_network(model);

    if(network == NULL) {
        log_error("%s", "Error creating network!");
        return;
    }
    ModelData* modelData = model->data;
    
    // default rate of keras -> 0.001
    // kaparthy's recommendation for adam: 0.0003
    double learningRate = 0.01;
    double currentLearningRate = learningRate;
    int step = 1;
    int totalEpochs = modelData->totalEpochs;

    network->optimizationConfig->learningRateDecayAmount = learningRate / totalEpochs;

    double minLoss = __DBL_MAX__;
    double maxAccuracy = 0.0;

      log_info("Starting training with learning rate of: %f for %d epochs.", learningRate, totalEpochs);
    while(step < totalEpochs) {
        // learningRates[step] = currentLearningRate;
        
        // for(int inputRow = 0; inputRow < model->data->trainingData->rows; inputRow++) {
        //     Vector* output = create_vector(network->layers[network->layerCount - 1]->neuronCount);
        //     forward_pass(network, modelData->trainingData->data[inputRow], output); 
        //     backpropagation(network, modelData->trainingData->data[inputRow], output, modelData->yValues_Training->data[inputRow]);
        //     network->output->data[inputRow] = copy_vector(output);
        //     free_vector(output);
        // }
        
        calculate_loss(network, modelData->yValues_Training);
        
        if(network->optimizationConfig->shouldUseLearningRateDecay == 1) {
            double decayRate = network->optimizationConfig->learningRateDecayAmount;
            currentLearningRate = currentLearningRate * (1 / (1.0 + (decayRate * (double)step)));
        }
        network->currentStep = step;
        network->optimizer(network, currentLearningRate);

        // if(step == 1 || step % 10 == 0){
            log_debug("Step: %d, Accuracy: %f, Loss: %f \n", step, network->accuracy, network->loss);  
        // }
        minLoss = fmin(minLoss, network->loss);
        
        maxAccuracy = fmax(maxAccuracy, network->accuracy);

        // losses[step] = network->loss;
        // storedSteps[step] = step;
        // accuracies[step] = network->accuracy;

        step++;
        // Clear the gradients
        for(int layerIndex = 0; layerIndex < network->layerCount; layerIndex++) {
            fill_matrix(network->layers[layerIndex]->gradients, 0.0f);
            fill_vector(network->layers[layerIndex]->biasGradients, 0.0f);
        }
    }

    log_info("Minimum loss during training: %f \n", minLoss);
    log_info("Maximum accuracy during training: %f \n", maxAccuracy);

    save_network(modelData->path, network);

    free_network(network);
}

void mnist_plot_data(ModelData* modelData) {
    // gnuplot_plot_xy(loss_step_plot, storedSteps, losses, totalEpochs, "Loss/Step");
    // gnuplot_plot_xy(accuracy_step_plot, storedSteps, accuracies, totalEpochs, "Accuracy/Step");
    // gnuplot_plot_xy(learningRate_step_plot, storedSteps, learningRates, totalEpochs, "Learning Rate/Step");

    // printf("Press enter to close the plots...\n");
    // getchar();
}

void mnist_free_plots(ModelData* modelData) {
    // free(losses);
    // free(storedSteps);
    // free(learningRates);
    // free(accuracies);

    // gnuplot_close(loss_step_plot);
    // gnuplot_close(accuracy_step_plot);
    // gnuplot_close(learningRate_step_plot);
}

void mnist_plot_config() { 
    // loss_step_plot = gnuplot_init();
    // learningRate_step_plot = gnuplot_init();
    // accuracy_step_plot = gnuplot_init();

    // gnuplot_setstyle(loss_step_plot, "linespoints");
    // gnuplot_set_xlabel(loss_step_plot, "Steps");
    // gnuplot_set_ylabel(loss_step_plot, "Loss");

    // gnuplot_setstyle(accuracy_step_plot, "linespoints");
    // gnuplot_set_xlabel(accuracy_step_plot, "Steps");
    // gnuplot_set_ylabel(accuracy_step_plot, "Loss");

    // gnuplot_setstyle(learningRate_step_plot, "dots");
    // gnuplot_set_xlabel(learningRate_step_plot, "Steps");
    // gnuplot_set_ylabel(learningRate_step_plot, "Learning Rate");
}

void mnist_validate_network(Model* model) {
    NNetwork* network = load_network(model->data->path);

    free_network(network);
}