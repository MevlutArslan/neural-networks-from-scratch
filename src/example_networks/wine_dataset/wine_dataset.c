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

    return model;
}

// doesnt seem to contain any leaks.
OptimizationConfig wine_categorization_create_optimizer(int optimizer) {
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


int wine_categorization_preprocess_data(ModelData* model_data) {
    Data* wine_data = load_csv("/Users/mevlutarslan/Downloads/datasets/wine_with_headers.csv");
    
    if(wine_data == NULL) {
        log_error("%s", "Failed to load Wine Data");
        return -1;
    }

    shuffle_rows(wine_data->data);

    // the amount we want to divide the dataset between training and validation data
    double divisionPercentage = 0.8;

    int targetColumn = 0;
    
    int trainingDataSize = wine_data->rows * divisionPercentage;

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

    config.activation_fns = calloc(config.numLayers, sizeof(enum ActivationFunction));
    
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

    train_network(network,model_data->training_data, model_data->training_labels, 0, model->data->total_epochs);

    // save_network(model_data->path, network);
    free_network(network);
}

/*
    While validating the network, it doesn't matter if you use row by row processing or batch processing as
    the only thing that matters is the weights and biases of the loaded network.
*/
void wine_categorization_validate_network(Model* model) {
    NNetwork* network = load_network(model->data->path);
    network->output = create_matrix(model->data->validation_data->rows, network->layers[network->num_layers - 1]->num_neurons);

    forward_pass_batched(network, model->data->validation_data);
    calculate_loss(network, model->data->validation_labels, network->batched_outputs[network->num_layers - 1]);

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

