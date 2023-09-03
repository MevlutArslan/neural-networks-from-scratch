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
    model->data->total_epochs = 2;
    model->data->loss_history = calloc(model->data->total_epochs, sizeof(double));
    model->data->epoch_history = calloc(model->data->total_epochs, sizeof(double));
    model->data->learning_rate_history = calloc(model->data->total_epochs, sizeof(double));
    model->data->accuracy_history = calloc(model->data->total_epochs, sizeof(double));
    model->data->path = "mnist_example_network";

    return model;
}

OptimizationConfig create_optimizer(int optimizer) {
    OptimizationConfig optimizationConfig;
    optimizationConfig.optimizer = optimizer;

    // Learning Rate Decay
    optimizationConfig.use_learning_rate_decay = 1;
    optimizationConfig.use_gradient_clipping = 0;

    // optimizationConfig.rho = 0.9;
    optimizationConfig.epsilon = 1e-8;
    optimizationConfig.adam_beta1 = 0.9;
    optimizationConfig.adam_beta2 = 0.999;

    return optimizationConfig;
}

NNetwork* mnist_get_network(Model* model) {
    if(model->preprocess_data(model->data) == -1) {
        log_error("%s", "Failed to complete preprocessing of MNIST data!");
    }

    NetworkConfig config;
    config.numLayers = 2;
    config.neurons_per_layer = malloc(sizeof(int) * config.numLayers);
    config.neurons_per_layer[0] = 128;
    config.neurons_per_layer[1] = 10;

    config.num_rows = model->data->training_data->rows;
    config.num_features = model->data->training_data->columns;

    OptimizationConfig optimizationConfig = create_optimizer(ADAM);

    // if you want to use l1 and/or l2 regularization you need to set the size to config.numLayers and 
    // fill these vectors with the lambda values you want
    config.weight_lambdas = create_vector(0);
    config.bias_lambdas = create_vector(0);

    if(config.weight_lambdas->size > 0 ){
        fill_vector(config.weight_lambdas, 1e-5);
    }

    if(config.bias_lambdas->size > 0 ){
        fill_vector(config.bias_lambdas, 1e-3);
    }

    config.activation_fns = calloc(config.numLayers, sizeof(enum ActivationFunction));  // Allocate memory
    
    config.optimization_config = malloc(sizeof(OptimizationConfig));
    memcpy(config.optimization_config, &optimizationConfig, sizeof(OptimizationConfig));

    for (int i = 0; i < config.numLayers - 1; i++) {
        config.activation_fns[i] = LEAKY_RELU;
    }

    // output layer's activation
    config.activation_fns[config.numLayers - 1] = SOFTMAX;

    config.loss_fn = CATEGORICAL_CROSS_ENTROPY;

    NNetwork* network = create_network(&config);

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

    modelData->training_data = get_sub_matrix_except_column(training_data->data, 0, trainingDataSize - 1, 0, training_data->columns - 1, 0);
    
    // extract validation data
    modelData->validation_data = get_sub_matrix_except_column(validation_data->data, 0, validation_data->rows - 1, 0, validation_data->columns - 1, 0);

    // extract yValues
    Vector* yValues_Training = extractYValues(training_data->data, 0);
    Vector* yValues_Testing = extractYValues(validation_data->data, 0);

    modelData->training_labels = oneHotEncode(yValues_Training, 10);
    modelData->validation_labels = oneHotEncode(yValues_Testing, 10);

    // normalize training data 
    for(int col = 0; col < modelData->training_data->columns; col++) {
        normalizeColumn_division(modelData->training_data, col, 255);
    }

    // normalize validation data
    for(int col = 0; col < modelData->validation_data->columns; col++) {
        normalizeColumn_division(modelData->validation_data, col, 255);
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
    ModelData* model_data = model->data;
    
    // default rate of keras -> 0.001
    // kaparthy's recommendation for adam: 0.0003
    double learningRate = 0.01;
    double currentLearningRate = learningRate;
    int epoch = 1;

    network->optimization_config->learning_rate_decay_amount = learningRate / model_data->total_epochs;

    double minLoss = __DBL_MAX__;
    double maxAccuracy = 0.0;

    log_info("Starting training with learning rate of: %f for %d epochs.", learningRate, model_data->total_epochs);
    while(epoch < model_data->total_epochs) {
        model_data->learning_rate_history[epoch] = currentLearningRate;
        
        for(int inputRow = 0; inputRow < model->data->training_data->rows; inputRow++) {
            Vector* output = create_vector(network->layers[network->num_layers - 1]->num_neurons);
            forward_pass_row_by_row(network, model_data->training_data->data[inputRow], output); 
            backpropagation(network, model_data->training_data->data[inputRow], output, model_data->training_labels->data[inputRow]);
            network->output->data[inputRow] = copy_vector(output);
            free_vector(output);
        }
        
        calculate_loss(network, model_data->validation_labels, network->output);
        
        if(network->optimization_config->use_learning_rate_decay == 1) {
            double decayRate = network->optimization_config->learning_rate_decay_amount;
            currentLearningRate = currentLearningRate * (1 / (1.0 + (decayRate * (double)epoch)));
        }
        network->training_epoch = epoch;
        network->optimization_algorithm(network, currentLearningRate, 0);

        // if(step == 1 || step % 10 == 0){
            log_debug("Epoch: %d, Accuracy: %f, Loss: %f \n", epoch, network->accuracy, network->loss);  
        // }
        minLoss = fmin(minLoss, network->loss);
        
        maxAccuracy = fmax(maxAccuracy, network->accuracy);

        // losses[step] = network->loss;
        // storedSteps[step] = step;
        // accuracies[step] = network->accuracy;

        epoch++;
        // Clear the gradients
        for(int layerIndex = 0; layerIndex < network->num_layers; layerIndex++) {
            fill_matrix(network->layers[layerIndex]->weight_gradients, 0.0f);
            fill_vector(network->layers[layerIndex]->bias_gradients, 0.0f);
        }
    }

    log_info("Minimum loss during training: %f \n", minLoss);
    log_info("Maximum accuracy during training: %f \n", maxAccuracy);

    save_network(model_data->path, network);

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