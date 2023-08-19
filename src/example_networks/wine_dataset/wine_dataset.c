#include "wine_dataset.h"

NNetwork* wine_categorization_get_network(Model* model);
void wine_categorization_train_network(Model* model);
void wine_categorization_validate_network(Model* model);
int wine_categorization_preprocess_data(ModelData* model_data);
void wine_categorization_plot_data(ModelData* model_data);
void wine_categorization_plot_config();

// doesnt seem to contain any leaks so far
Model* create_wine_categorization_model() {
    Model* model = malloc(sizeof(Model));

    model->get_network = wine_categorization_get_network;
    model->train_network = wine_categorization_train_network;
    model->validate_network = wine_categorization_validate_network;
    model->preprocess_data = wine_categorization_preprocess_data;
    model->plot_data = wine_categorization_plot_data;
    model->plot_config = wine_categorization_plot_config;


    model->data = (ModelData*) malloc(sizeof(ModelData));
    model->data->total_epochs = 250;
    model->data->loss_history = calloc(model->data->total_epochs, sizeof(double));
    model->data->epoch_history = calloc(model->data->total_epochs, sizeof(double));
    model->data->learning_rate_history = calloc(model->data->total_epochs, sizeof(double));
    model->data->accuracy_history = calloc(model->data->total_epochs, sizeof(double));
    model->data->path = "wine_dataset_network";

    // model->thread_pool = create_thread_pool(6);

    return model;
}

// doesnt seem to contain any leaks.
OptimizationConfig wine_categorization_create_optimizer(int optimizer) {
    OptimizationConfig optimization_config;
    optimization_config.optimizer = optimizer;

    // Learning Rate Decay
    optimization_config.use_learning_rate_decay = 1;
    optimization_config.use_gradient_clipping = 0;
    // optimizationConfig.rho = 0.9;
    optimization_config.epsilon = 1e-8;
    optimization_config.adam_beta1 = 0.9;
    optimization_config.adam_beta2 = 0.999;

    return optimization_config;
}


int wine_categorization_preprocess_data(ModelData* model_data) {
    Data* wine_data = load_csv("/home/mvlcfr/datasets/wine_dataset/wine_with_headers.csv");
    
    if(wine_data == NULL) {
        log_error("%s", "Failed to load Wine Data");
        return -1;
    }

    shuffle_rows(wine_data->data);
    // log_info("shuffled rows: %s", matrix_to_string(wineData->data));
    // the amount we want to divide the dataset between training and validation data
    double divisionPercentage = 0.8;

    int targetColumn = 0;
    
    int trainingDataSize = wine_data->rows * divisionPercentage;

    // TODO: OPTIMIZE
    Matrix* trainingData = get_sub_matrix(wine_data->data, 0, trainingDataSize - 1, 0, wine_data->data->columns - 1);
    model_data->training_data = get_sub_matrix_except_column(trainingData, 0, trainingData->rows - 1, 0, trainingData->columns - 1, targetColumn);
    
    Vector* yValues_training = extractYValues(trainingData, targetColumn);
    model_data->training_labels = oneHotEncode(yValues_training, 3);

    
    for(int colIndex = 0; colIndex < model_data->training_data->columns; colIndex++) {
        normalizeColumn_standard_deviation(model_data->training_data, colIndex);
    }
    
    Matrix* validation_data = get_sub_matrix(wine_data->data, model_data->training_data->rows, wine_data->rows - 1, 0, wine_data->data->columns - 1);
    model_data->validation_data = get_sub_matrix_except_column(validation_data, 0, validation_data->rows -1, 0, validation_data->columns - 1, targetColumn);

    Vector* validation_labels = extractYValues(validation_data, targetColumn);
    model_data->validation_labels = oneHotEncode(validation_labels, 3);
    
    for(int colIndex = 0; colIndex < model_data->validation_data->columns; colIndex++) {
        normalizeColumn_standard_deviation(model_data->validation_data, colIndex);
    }

    free_matrix(trainingData);
    free_matrix(validation_data);
    free_vector(yValues_training);
    free_vector(validation_labels);
    free_data(wine_data);

    return 1;
}


NNetwork* wine_categorization_get_network(Model* model) {
    if(model->preprocess_data(model->data) != 1) {
        log_error("%s", "Failed to complete preprocessing of Wine Categorization data!");
    }
    NetworkConfig config;
    config.numLayers = 2;
    config.neurons_per_layer = malloc(sizeof(int) * config.numLayers);
    config.neurons_per_layer[0] = 2;
    config.neurons_per_layer[1] = 3;

    config.num_rows = model->data->training_data->rows;
    config.num_features = model->data->training_data->columns;

    OptimizationConfig optimizationConfig = wine_categorization_create_optimizer(ADAM);

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

    config.activation_fns = calloc(config.numLayers, sizeof(ActivationFunction));
    
    config.optimization_config = malloc(sizeof(OptimizationConfig));
    memcpy(config.optimization_config, &optimizationConfig, sizeof(OptimizationConfig));
    
    for (int i = 0; i < config.numLayers - 1; i++) {
        memcpy(&config.activation_fns[i], &LEAKY_RELU, sizeof(ActivationFunction));
    }

    // output layer's activation
    memcpy(&config.activation_fns[config.numLayers - 1], &SOFTMAX, sizeof(ActivationFunction));

    config.loss_fn = malloc(sizeof(LossFunction));
    memcpy(&config.loss_fn->loss_function, &CATEGORICAL_CROSS_ENTROPY, sizeof(LossFunction));

    NNetwork* network = create_network(&config);

    free_network_config(&config);
    model->plot_config();

    return network;
}

void wine_categorization_train_network(Model* model) {
    NNetwork* network = wine_categorization_get_network(model);

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

    log_debug("Starting training with learning rate of: %f for %d epochs.", learningRate,  model_data->total_epochs);
    while(epoch < model_data->total_epochs) {
        model_data->learning_rate_history[epoch] = currentLearningRate;

        forward_pass_batched(network, model_data->training_data); 
        backpropagation_batched(network, model_data->training_data, model_data->training_labels);

        calculate_loss(network, model_data->training_labels);

        if(network->optimization_config->use_learning_rate_decay == 1) {
            double decayRate = network->optimization_config->learning_rate_decay_amount;
            currentLearningRate = currentLearningRate * (1 / (1.0 + (decayRate * (double)epoch)));
        }

        network->training_epoch = epoch;
        network->optimization_algorithm(network, currentLearningRate);

        if(epoch == 1 || epoch % 10 == 0){
            log_debug("Epoch: %d, Accuracy: %f, Loss: %f \n", epoch, network->accuracy, network->loss); 
        }
        minLoss = fmin(minLoss, network->loss);
        
        maxAccuracy = fmax(maxAccuracy, network->accuracy);

        model_data->loss_history[epoch] = network->loss;
        model_data->epoch_history[epoch] = epoch;
        model_data->accuracy_history[epoch] = network->accuracy;
        epoch++;
        // Clear the gradients
        for(int layerIndex = 0; layerIndex < network->num_layers; layerIndex++) {
            fill_matrix(network->weight_gradients[layerIndex], 0.0f);
            fill_vector(network->bias_gradients[layerIndex], 0.0f);
        }
    }

    log_info("Minimum loss during training: %f \n", minLoss);
    log_info("Maximum accuracy during training: %f \n", maxAccuracy);
    
    save_network(model_data->path, network);

    // model->plot_data(model->data);
    free_network(network);
}

void wine_categorization_validate_network(Model* model) {
    NNetwork* network = load_network(model->data->path);
    
    init_network_memory(network, model->data->validation_data->rows);

    forward_pass_batched(network, model->data->validation_data);
    calculate_loss(network, model->data->validation_labels);

    log_info("Validation Loss: %f", network->loss);
    log_info("Validation Accuracy: %f", network->accuracy);

    free_network(network);
}

gnuplot_ctrl* loss_step_plot;
gnuplot_ctrl* accuracy_step_plot;
gnuplot_ctrl* learning_rate_step_plot;
void wine_categorization_plot_data(ModelData* model_data) {
    gnuplot_plot_xy(loss_step_plot, model_data->epoch_history, model_data->loss_history, model_data->total_epochs, "loss/step");
    gnuplot_plot_xy(accuracy_step_plot, model_data->epoch_history, model_data->accuracy_history, model_data->total_epochs, "accuracy/step");
    gnuplot_plot_xy(learning_rate_step_plot, model_data->epoch_history, model_data->learning_rate_history, model_data->total_epochs, "learning rate/step");

    printf("press Enter to close the plots!");
    getchar();

    gnuplot_close(loss_step_plot);
    gnuplot_close(accuracy_step_plot);
    gnuplot_close(learning_rate_step_plot);
}

void wine_categorization_plot_config() {
    loss_step_plot = gnuplot_init();

    gnuplot_set_xlabel(loss_step_plot, "step");
    gnuplot_set_ylabel(loss_step_plot, "loss");
    
    accuracy_step_plot = gnuplot_init();
    
    gnuplot_set_xlabel(accuracy_step_plot, "step");
    gnuplot_set_ylabel(accuracy_step_plot, "accuracy");

    learning_rate_step_plot = gnuplot_init();
    
    gnuplot_set_xlabel(learning_rate_step_plot, "step");
    gnuplot_set_ylabel(learning_rate_step_plot, "learning rate");

}

