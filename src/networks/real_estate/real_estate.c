#include "../model.h"

Model* create_real_estate_model() {
    Model* model = (Model*)malloc(sizeof(Model));
    assert(model != NULL);

    model->get_network = real_estate_get_network;
    model->preprocess_data = real_estate_preprocess_data;

    model->data = (ModelData*) malloc(sizeof(ModelData));
    assert(model->data != NULL);

    model->data->total_epochs = 500;
    model->data->loss_history = calloc(model->data->total_epochs, sizeof(double));
    model->data->epoch_history = calloc(model->data->total_epochs, sizeof(double));
    model->data->learning_rate_history = calloc(model->data->total_epochs, sizeof(double));
    model->data->accuracy_history = calloc(model->data->total_epochs, sizeof(double));
    
    model->data->save_path = "real_estate_network";

    // TODO: Implement ability to divide the dataset into multiple batches.
    model->data->num_batches = 1;

    return model;
}

void real_estate_preprocess_data(ModelData* model_data) {
    Data* real_estate_data = load_csv("/Users/mevlutarslan/Downloads/datasets/Real estate.csv");
    assert(real_estate_data != NULL);

    shuffle_rows(real_estate_data->data);

    float separation_factor = 0.7;

    int training_data_num_rows = real_estate_data->rows * separation_factor;
    // separate training and validation

    // last column is the y value (expected value) so I omitted it out of the training values
    model_data->training_data = get_sub_matrix(real_estate_data->data, 0, training_data_num_rows, 1, real_estate_data->columns - 2);
    assert(model_data->training_data != NULL);
    // log_info("training data: %s", matrix_to_string(model_data->training_data));

    // get the last column and store it in a matrix of dimensions (training_data_num_rows x 1)
    model_data->training_labels = get_sub_matrix(real_estate_data->data, 0, training_data_num_rows, real_estate_data->columns - 1, real_estate_data->columns - 1);
    assert(model_data->training_labels != NULL);
    // log_info("training labels: %s", matrix_to_string(model_data->training_labels));
    
    model_data->validation_data = get_sub_matrix(real_estate_data->data, training_data_num_rows + 1, real_estate_data->rows - 1, 1, real_estate_data->columns -  2);
    assert(model_data->validation_data != NULL);
    // log_info("validation data: %s", matrix_to_string(model_data->validation_data));
    
    model_data->validation_labels = get_sub_matrix(real_estate_data->data, training_data_num_rows + 1, real_estate_data->rows - 1, real_estate_data->columns - 1, real_estate_data->columns -  1);
    assert(model_data->validation_labels != NULL);
    // log_info("validation labels: %s", matrix_to_string(model_data->validation_labels));

    for(int i = 0; i < model_data->training_data->columns; i++) {
        normalize_column(STANDARD_DEVIATION, model_data->training_data, i, 0);
    }

    for(int i = 0; i < model_data->validation_data->columns; i++) {
        normalize_column(STANDARD_DEVIATION, model_data->validation_data, i, 0);
    }
}

NNetwork* real_estate_get_network(Model* model) {
    real_estate_preprocess_data(model->data);

    NetworkConfig network_config;
    network_config.num_layers = 2;
    network_config.neurons_per_layer = (int*) calloc(network_config.num_layers, sizeof(int));
    network_config.neurons_per_layer[0] = 10;
    network_config.neurons_per_layer[1] = 1;


    network_config.num_features = model->data->training_data->columns;
    network_config.num_rows = model->data->training_data->rows;

    network_config.activation_fns = (ActivationFunction*) calloc(network_config.num_layers, sizeof(enum ActivationFunction));
    network_config.activation_fns[0] = LEAKY_RELU;
    network_config.activation_fns[1] = LEAKY_RELU;

    network_config.loss_fn = MEAN_SQUARED_ERROR;

    OptimizationConfig* optimization_config = (OptimizationConfig*) calloc(1, sizeof(OptimizationConfig));
    assert(optimization_config != NULL);
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

    network_config.optimization_config = optimization_config;

    NNetwork* network = create_network(&network_config);
    assert(network != NULL);

    log_info("%s", "Created Network:");
    dump_network_config(network);

    sleep(1);

    return network;
}

