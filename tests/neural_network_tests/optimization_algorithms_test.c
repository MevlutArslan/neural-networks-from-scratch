#include "optimization_algorithms_test.h"

NNetwork* create_mock_network(OptimizationConfig* optimization_config) {
    NetworkConfig network_config;
    network_config.numLayers = 2;
    
    network_config.neurons_per_layer = (int*) calloc(network_config.numLayers, sizeof(int));
    network_config.neurons_per_layer[0] = 3;
    network_config.neurons_per_layer[1] = 2;

    network_config.activation_fns = calloc(network_config.numLayers, sizeof(enum ActivationFunction));
    network_config.optimization_config = optimization_config;

    for(int i = 0; i < network_config.numLayers - 1; i++) {
        network_config.activation_fns[i] = LEAKY_RELU;
    }

    network_config.activation_fns[network_config.numLayers - 1] = SOFTMAX;

    network_config.learning_rate = 0.01;
    network_config.num_rows = 1;
    network_config.num_features = 4;

    network_config.weight_lambdas = create_vector(0);
    network_config.bias_lambdas = create_vector(0);

    network_config.loss_fn = CATEGORICAL_CROSS_ENTROPY;

    NNetwork* mock_network = create_network(&network_config);
    assert(mock_network != NULL);

    return mock_network;
}

void test_mock_sgd() {
    OptimizationConfig optimization_config;
    optimization_config.optimizer = SGD;

    NNetwork* mock_network = create_mock_network(&optimization_config);

    double layer_0_gradient = 0.5f;
    double layer_1_gradient = 0.5f;
    fill_matrix(mock_network->layers[0]->weight_gradients, layer_0_gradient);
    fill_vector(mock_network->layers[0]->bias_gradients, layer_0_gradient);

    fill_matrix(mock_network->layers[1]->weight_gradients, layer_1_gradient);
    fill_vector(mock_network->layers[1]->bias_gradients, layer_1_gradient);

    double learning_rate = 0.01;
    double factor = learning_rate * layer_0_gradient;

    // sgd is without momentum is weights -= learning rate * gradient
    Matrix* expected_layer_0 = create_matrix(mock_network->layers[0]->weights->rows, mock_network->layers[0]->weights->columns);
    for(int i = 0; i < expected_layer_0->rows; i++) {
        for(int j = 0; j < expected_layer_0->columns; j++) {
            expected_layer_0->data[i]->elements[j] = mock_network->layers[0]->weights->data[i]->elements[j] - factor;
        }
    }

    Vector* expected_layer_0_bias = create_vector(mock_network->layers[0]->num_neurons);
    for(int i = 0; i < expected_layer_0_bias->size; i++) {
        expected_layer_0_bias->elements[i] = mock_network->layers[0]->biases->elements[i] - factor;
    }

    sgd(mock_network, 0.01, 0);

    assert(is_equal_matrix(expected_layer_0, mock_network->layers[0]->weights) == 1);
    assert(is_equal_vector(expected_layer_0_bias, mock_network->layers[0]->biases) == 1);
}

void test_mock_adagrad() {

}

void test_mock_rms_prop() {

}

void test_mock_adam() {

}