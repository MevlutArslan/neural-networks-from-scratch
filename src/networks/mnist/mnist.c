#include "../model.h"


Model* create_mnist_model() {
    
    Model* model = malloc(sizeof(Model));

    model->get_network = &mnist_get_network;
    model->preprocess_data = &mnist_preprocess_data;


    model->data = (ModelData*) malloc(sizeof(ModelData));
    model->data->total_epochs = 10;
    model->data->loss_history = calloc(model->data->total_epochs, sizeof(double));
    model->data->epoch_history = calloc(model->data->total_epochs, sizeof(double));
    model->data->learning_rate_history = calloc(model->data->total_epochs, sizeof(double));
    model->data->accuracy_history = calloc(model->data->total_epochs, sizeof(double));

    model->data->num_batches = 0; // for sequential processing.
    
    model->data->save_path = "mnist_example_network";

    return model;
}

void mnist_preprocess_data(ModelData* model_data) {
    Data* training_data = load_csv("/Users/mevlutarslan/Downloads/datasets/mnistcsv/mnist_train.csv");
    Data* validation_data = load_csv("/Users/mevlutarslan/Downloads/datasets/mnistcsv/mnist_test.csv");

    assert(training_data != NULL);
    assert(validation_data != NULL);

    // extract training data
    int targetColumn = 0;
    int trainingDataSize = training_data->rows;

    model_data->training_data = get_sub_matrix_except_column(training_data->data, 0, trainingDataSize - 1, 0, training_data->columns - 1, 0);
    
    // extract validation data
    model_data->validation_data = get_sub_matrix_except_column(validation_data->data, 0, validation_data->rows - 1, 0, validation_data->columns - 1, 0);

    // extract yValues
    Vector* yValues_Training = extractYValues(training_data->data, 0);
    Vector* yValues_Testing = extractYValues(validation_data->data, 0);

    model_data->training_labels = oneHotEncode(yValues_Training, 10);
    model_data->validation_labels = oneHotEncode(yValues_Testing, 10);

    // normalize training data 
    for(int col = 0; col < model_data->training_data->columns; col++) {
        normalize_column(BY_DIVISION, model_data->training_data, col, 255);
    }

    // normalize validation data
    for(int col = 0; col < model_data->validation_data->columns; col++) {
        normalize_column(BY_DIVISION, model_data->validation_data, col, 255);
    }
    
    free_data(training_data);
    free_data(validation_data);
    free_vector(yValues_Training);
    free_vector(yValues_Testing);
}


NNetwork* mnist_get_network(Model* model) {
    model->preprocess_data(model->data);

    NetworkConfig config;
    config.numLayers = 2;
    config.neurons_per_layer = malloc(sizeof(int) * config.numLayers);
    config.neurons_per_layer[0] = 128;
    config.neurons_per_layer[1] = 10;

    config.num_rows = model->data->training_data->rows;
    config.num_features = model->data->training_data->columns;


    config.activation_fns = calloc(config.numLayers, sizeof(enum ActivationFunction));  // Allocate memory

    for (int i = 0; i < config.numLayers - 1; i++) {
        config.activation_fns[i] = LEAKY_RELU;
    }

    // output layer's activation
    config.activation_fns[config.numLayers - 1] = SOFTMAX;

    config.loss_fn = CATEGORICAL_CROSS_ENTROPY;


    OptimizationConfig* optimization_config = (OptimizationConfig*) calloc(1, sizeof(OptimizationConfig));
    optimization_config->optimizer = ADAM;

    // Learning Rate Decay
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

    dump_network_config(network);

    return network;
}