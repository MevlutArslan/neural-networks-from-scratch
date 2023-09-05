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
    Matrix* expected_updated_weights = create_matrix(mock_network->layers[0]->weights->rows, mock_network->layers[0]->weights->columns);
    for(int i = 0; i < expected_updated_weights->rows; i++) {
        for(int j = 0; j < expected_updated_weights->columns; j++) {
            expected_updated_weights->data[i]->elements[j] = mock_network->layers[0]->weights->data[i]->elements[j] - factor;
        }
    }

    Vector* expected_updated_biases = create_vector(mock_network->layers[0]->num_neurons);
    for(int i = 0; i < expected_updated_biases->size; i++) {
        expected_updated_biases->elements[i] = mock_network->layers[0]->biases->elements[i] - factor;
    }

    sgd(mock_network, 0.01, 0);

    assert(is_equal_matrix(expected_updated_weights, mock_network->layers[0]->weights) == 1);
    assert(is_equal_vector(expected_updated_biases, mock_network->layers[0]->biases) == 1);

    free_matrix(expected_updated_weights);
    free_vector(expected_updated_biases);

    free_network(mock_network);
}

void test_mock_adagrad() {
    OptimizationConfig optimization_config;
    optimization_config.optimizer = ADAGRAD;
    optimization_config.epsilon = 1e-8; // Set the epsilon value

    NNetwork* mock_network = create_mock_network(&optimization_config);

    double layer_0_gradient = 0.5;
    double layer_1_gradient = 0.5;
    fill_matrix(mock_network->layers[0]->weight_gradients, layer_0_gradient);
    fill_vector(mock_network->layers[0]->bias_gradients, layer_0_gradient);

    fill_matrix(mock_network->layers[1]->weight_gradients, layer_1_gradient);
    fill_vector(mock_network->layers[1]->bias_gradients, layer_1_gradient);

    double learning_rate = 0.01;

    // Create matrices and vectors to accumulate expected updated values and cache values
    Matrix* expected_updated_weights = create_matrix(mock_network->layers[0]->weights->rows, mock_network->layers[0]->weights->columns);
    Vector* expected_updated_biases = create_vector(mock_network->layers[0]->biases->size);
    
    Matrix* accumulated_weight_cache = create_matrix(mock_network->layers[0]->weights->rows, mock_network->layers[0]->weights->columns);
    Vector* accumulated_bias_cache = create_vector(mock_network->layers[0]->biases->size);

    // Calculate the expected updated values and accumulate them
    Layer* current_layer = mock_network->layers[0];

    for(int neuron_index = 0; neuron_index < current_layer->num_neurons; neuron_index++) {
        for(int weight_index = 0; weight_index < current_layer->weights->columns; weight_index++) {
            double weight_gradient = current_layer->weight_gradients->data[neuron_index]->elements[weight_index];

            double weight_update = learning_rate * weight_gradient;
            double gradient_squared = weight_gradient * weight_gradient;

            accumulated_weight_cache->data[neuron_index]->elements[weight_index] += gradient_squared;
            expected_updated_weights->data[neuron_index]->elements[weight_index] = current_layer->weights->data[neuron_index]->elements[weight_index] - weight_update / (sqrt(accumulated_weight_cache->data[neuron_index]->elements[weight_index]) + mock_network->optimization_config->epsilon);
        }

        double bias_gradient = current_layer->bias_gradients->elements[neuron_index];

        double bias_update = learning_rate * bias_gradient;
        double gradient_squared = bias_gradient * bias_gradient;

        accumulated_bias_cache->elements[neuron_index] += gradient_squared;
        expected_updated_biases->elements[neuron_index] = current_layer->biases->elements[neuron_index] - bias_update / (sqrt(accumulated_bias_cache->elements[neuron_index]) + mock_network->optimization_config->epsilon);
    }
    
    // Call the Adagrad function
    adagrad(mock_network, learning_rate, 0);

    // Compare the actual updated weights and biases with the expected accumulated values
    assert(is_equal_matrix(expected_updated_weights, mock_network->layers[0]->weights) == 1);
    assert(is_equal_vector(expected_updated_biases, mock_network->layers[0]->biases) == 1);

    // Compare the accumulated cache values with the actual cache values
    assert(is_equal_matrix(accumulated_weight_cache, mock_network->layers[0]->weight_cache) == 1);
    assert(is_equal_vector(accumulated_bias_cache, mock_network->layers[0]->bias_cache) == 1);

    free_matrix(expected_updated_weights);
    free_matrix(accumulated_weight_cache);

    free_vector(expected_updated_biases);
    free_vector(accumulated_bias_cache);

    free_network(mock_network);
}

void test_mock_rms_prop() {

}

void test_mock_adam() {

}