#include "../model.h"

Model* create_wine_categorization_model() {
    Model* model = malloc(sizeof(Model));

    model->get_network = wine_categorization_get_network;
    model->preprocess_data = wine_categorization_preprocess_data;

    model->data = (ModelData*) malloc(sizeof(ModelData));
    model->data->total_epochs = 250;
    model->data->loss_history = calloc(model->data->total_epochs, sizeof(double));
    model->data->epoch_history = calloc(model->data->total_epochs, sizeof(double));
    model->data->learning_rate_history = calloc(model->data->total_epochs, sizeof(double));
    model->data->accuracy_history = calloc(model->data->total_epochs, sizeof(double));
    
    model->data->save_path = "wine_dataset_network";

    // TODO: Implement ability to divide the dataset into multiple batches.
    model->data->num_batches = 1;

    return model;
}

void wine_categorization_preprocess_data(ModelData* model_data) {
    Data* wine_data = load_csv("/Users/mevlutarslan/Downloads/datasets/wine_with_headers.csv");
    assert(wine_data != NULL);

    shuffle_rows(wine_data->data);

    // the amount we want to divide the dataset between training and validation data
    double divisionPercentage = 0.8;

    int targetColumn = 0;
    
    int trainingDataSize = wine_data->rows * divisionPercentage;

    model_data->training_data = get_sub_matrix(wine_data->data, 0, trainingDataSize - 1, 1, wine_data->data->columns - 1);
    // log_info("training data: %s", matrix_to_string(model_data->training_data));

    Matrix* training_data_with_target = get_sub_matrix(wine_data->data, 0, trainingDataSize - 1, 0, wine_data->data->columns - 1);
    
    Vector* training_labels = extractYValues(training_data_with_target, targetColumn);
    model_data->training_labels = oneHotEncode(training_labels, 3);
    
    model_data->validation_data = get_sub_matrix(wine_data->data, model_data->training_data->rows, wine_data->rows - 1, 1, wine_data->data->columns - 1);

    Matrix* validation_data_with_targets = get_sub_matrix(wine_data->data, model_data->training_data->rows, wine_data->rows - 1, 0, wine_data->data->columns - 1);
    
    Vector* validation_labels = extractYValues(validation_data_with_targets, targetColumn);
    model_data->validation_labels = oneHotEncode(validation_labels, 3);
    
    for(int colIndex = 0; colIndex < model_data->training_data->columns; colIndex++) {
        normalize_column(STANDARD_DEVIATION, model_data->training_data, colIndex, 0);
    }

    for(int colIndex = 0; colIndex < model_data->validation_data->columns; colIndex++) {
        normalize_column(STANDARD_DEVIATION, model_data->validation_data, colIndex, 0);
    }

    free_matrix(training_data_with_target);
    free_matrix(validation_data_with_targets);
    free_vector(training_labels);
    free_vector(validation_labels);
    free_data(wine_data);
}


NNetwork* wine_categorization_get_network(Model* model) {
    model->preprocess_data(model->data);

    NetworkConfig config;
    config.num_layers = 2;
    config.neurons_per_layer = (int*) calloc(config.num_layers, sizeof(int));
    config.neurons_per_layer[0] = 2;
    config.neurons_per_layer[1] = 3;

    config.num_rows = model->data->training_data->rows;
    config.num_features = model->data->training_data->columns;

    config.activation_fns = calloc(config.num_layers, sizeof(enum ActivationFunction));
    
    for (int i = 0; i < config.num_layers - 1; i++) {
        config.activation_fns[i] = LEAKY_RELU;
    }

    // output layer's activation
    config.activation_fns[config.num_layers - 1] = SOFTMAX;
    
    config.loss_fn = CATEGORICAL_CROSS_ENTROPY;

    OptimizationConfig* optimization_config = (OptimizationConfig*) calloc(1, sizeof(OptimizationConfig));
    optimization_config->optimizer = ADAM;
    optimization_config->learning_rate = 0.01;

    optimization_config->use_learning_rate_decay = 1;
    optimization_config->use_gradient_clipping = 0;
    optimization_config->gradient_clip_lower_bound = 0;
    optimization_config->gradient_clip_upper_bound = 0;
    optimization_config->use_momentum = 0;

    optimization_config->rho = 0;
    optimization_config->epsilon = 1e-8;
    optimization_config->adam_beta1 = 0.9;
    optimization_config->adam_beta2 = 0.999;
    
    // if you want to apply regularization:
    /*
        optimization_config->use_l1_regularization = TRUE;
        optimization_config->use_l2_regularization = TRUE;  

        optimization_config->l1_weight_lambdas = vector
        optimization_config->l2_weight_lambdas = vector

        optimization_config->l1_bias_lambdas = vector
        optimization_config->l2_bias_lambdas = vector
    */
    
    config.optimization_config = optimization_config;

    NNetwork* network = create_network(&config);
    assert(network != NULL);

    log_info("%s", "Created Network:");
    dump_network_config(network);

    return network;
}