#include "optimization_algorithms_test.h"

NNetwork* create_mock_network(OptimizationConfig* optimization_config) {
    NetworkConfig network_config;
    network_config.num_layers = 2;
    
    network_config.neurons_per_layer = (int*) calloc(network_config.num_layers, sizeof(int));
    network_config.neurons_per_layer[0] = 3;
    network_config.neurons_per_layer[1] = 2;

    network_config.activation_fns = calloc(network_config.num_layers, sizeof(enum ActivationFunction));
    network_config.optimization_config = optimization_config;

    for(int i = 0; i < network_config.num_layers - 1; i++) {
        network_config.activation_fns[i] = LEAKY_RELU;
    }

    network_config.activation_fns[network_config.num_layers - 1] = SOFTMAX;

    network_config.learning_rate = 0.01;
    network_config.num_rows = 1;
    network_config.num_features = 4;


    network_config.loss_fn = CATEGORICAL_CROSS_ENTROPY;

    NNetwork* mock_network = create_network(&network_config);
    assert(mock_network != NULL);

    return mock_network;
}

void test_mock_sgd() {
    OptimizationConfig optimization_config;
    optimization_config.optimizer = SGD;

    optimization_config.use_l1_regularization = FALSE;
    optimization_config.use_l2_regularization = FALSE;

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

    log_info("SGD test passed successfully.");
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

    // Create matrices and vectors to accumulate expected updated values
    Matrix* expected_updated_weights = create_matrix(mock_network->layers[0]->weights->rows, mock_network->layers[0]->weights->columns);
    Vector* expected_updated_biases = create_vector(mock_network->layers[0]->biases->size);
    
    // Calculate the expected updated values and accumulate them
    Layer* current_layer = mock_network->layers[0];

    for(int neuron_index = 0; neuron_index < current_layer->num_neurons; neuron_index++) {
        for(int weight_index = 0; weight_index < current_layer->weights->columns; weight_index++) {
            double weight_gradient = current_layer->weight_gradients->data[neuron_index]->elements[weight_index];

            double weight_update = learning_rate * weight_gradient;
            double gradient_squared = weight_gradient * weight_gradient;

            current_layer->weight_cache->data[neuron_index]->elements[weight_index] += gradient_squared;
            expected_updated_weights->data[neuron_index]->elements[weight_index] = current_layer->weights->data[neuron_index]->elements[weight_index] - weight_update / (sqrt(current_layer->weight_cache->data[neuron_index]->elements[weight_index]) + mock_network->optimization_config->epsilon);
        }

        double bias_gradient = current_layer->bias_gradients->elements[neuron_index];

        double bias_update = learning_rate * bias_gradient;
        double gradient_squared = bias_gradient * bias_gradient;

        current_layer->bias_cache->elements[neuron_index] += gradient_squared;
        expected_updated_biases->elements[neuron_index] = current_layer->biases->elements[neuron_index] - bias_update / (sqrt(current_layer->bias_cache->elements[neuron_index]) + mock_network->optimization_config->epsilon);
    }
    
    // Call the Adagrad function
    adagrad(mock_network, learning_rate, 0);

    // Compare the actual updated weights and biases with the expected accumulated values
    assert(is_equal_matrix(expected_updated_weights, mock_network->layers[0]->weights) == 1);
    assert(is_equal_vector(expected_updated_biases, mock_network->layers[0]->biases) == 1);

    free_matrix(expected_updated_weights);
    free_vector(expected_updated_biases);

    free_network(mock_network);

    log_info("Adagrad test passed successfully.");
}

void test_mock_rms_prop() {
    OptimizationConfig optimization_config;
    optimization_config.optimizer = RMS_PROP;
    optimization_config.epsilon = 1e-8; // Set the epsilon value
    optimization_config.rho = 0.9; // Set the rho value

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

    // Calculate the expected updated values and accumulate them
    Layer* current_layer = mock_network->layers[0];

    for(int neuron_index = 0; neuron_index < current_layer->num_neurons; neuron_index++) {
        for(int weight_index = 0; weight_index < current_layer->weights->columns; weight_index++) {
            double weight_gradient = current_layer->weight_gradients->data[neuron_index]->elements[weight_index];

            double weight_update = -1 * (learning_rate * weight_gradient);
            double gradient_squared = weight_gradient * weight_gradient;

            // cache = rho * cache + (1 - rho) * weight_gradient ** 2
            double cache_fraction = optimization_config.rho * current_layer->weight_cache->data[neuron_index]->elements[weight_index];
            double gradient_squared_fraction = (1 - optimization_config.rho) * gradient_squared;

            current_layer->weight_cache->data[neuron_index]->elements[weight_index] += gradient_squared_fraction;
            expected_updated_weights->data[neuron_index]->elements[weight_index] = current_layer->weights->data[neuron_index]->elements[weight_index] + weight_update / (sqrt(cache_fraction + gradient_squared_fraction) + mock_network->optimization_config->epsilon);
        }

        double bias_gradient = current_layer->bias_gradients->elements[neuron_index];

        double bias_update = -1 * (learning_rate * bias_gradient);
        double gradient_squared = bias_gradient * bias_gradient;

        //  cache = rho * cache + (1 - rho) * gradient ** 2
        double cache_fraction = optimization_config.rho * current_layer->bias_cache->elements[neuron_index];
        double gradient_squared_fraction = (1 - optimization_config.rho) * gradient_squared;

        current_layer->bias_cache->elements[neuron_index] += gradient_squared_fraction;
        expected_updated_biases->elements[neuron_index] = current_layer->biases->elements[neuron_index] + bias_update / (sqrt(cache_fraction + gradient_squared_fraction) + mock_network->optimization_config->epsilon);
    }
    
    // Call the RMSprop function
    rms_prop(mock_network, learning_rate, 0);

    // Compare the actual updated weights and biases with the expected accumulated values
    assert(is_equal_matrix(expected_updated_weights, mock_network->layers[0]->weights) == 1);
    assert(is_equal_vector(expected_updated_biases, mock_network->layers[0]->biases) == 1);

    free_matrix(expected_updated_weights);
    free_vector(expected_updated_biases);

    free_network(mock_network);

    log_info("RMSprop test passed successfully.");
}

void test_mock_adam() {
    OptimizationConfig optimization_config;
    optimization_config.optimizer = ADAM;
    optimization_config.epsilon = 1e-8; // Set the epsilon value
    optimization_config.adam_beta1 = 0.9; // Set the beta1 value
    optimization_config.adam_beta2 = 0.999; // Set the beta2 value

    NNetwork* mock_network = create_mock_network(&optimization_config);

    double layer_0_gradient = 0.5;
    double layer_1_gradient = 0.5;
    fill_matrix(mock_network->layers[0]->weight_gradients, layer_0_gradient);
    fill_vector(mock_network->layers[0]->bias_gradients, layer_0_gradient);

    fill_matrix(mock_network->layers[1]->weight_gradients, layer_1_gradient);
    fill_vector(mock_network->layers[1]->bias_gradients, layer_1_gradient);

    double learning_rate = 0.01;
    int simulated_training_epoch = 1;

    // Create matrices and vectors to accumulate expected updated values and cache values
    Matrix* expected_updated_weights = create_matrix(mock_network->layers[0]->weights->rows, mock_network->layers[0]->weights->columns);
    Vector* expected_updated_biases = create_vector(mock_network->layers[0]->biases->size);
    
    double beta1 = mock_network->optimization_config->adam_beta1;
    double beta2 = mock_network->optimization_config->adam_beta2;

    double epsilon = mock_network->optimization_config->epsilon;

    // Calculate the expected updated values and accumulate them
    Layer* current_layer = mock_network->layers[0];

    for(int neuron_index = 0; neuron_index < current_layer->num_neurons; neuron_index++) {
        for(int weight_index = 0; weight_index < current_layer->weights->columns; weight_index++) {
            double weight_gradient = current_layer->weight_gradients->data[neuron_index]->elements[weight_index];
                
            // m(t) = beta1 * m(t-1) + (1 – beta1) * g(t)
            double old_momentum =  current_layer->weight_momentums->data[neuron_index]->elements[weight_index];
            double weight_momentum = beta1 * old_momentum + (1 - beta1) * weight_gradient;

            current_layer->weight_momentums->data[neuron_index]->elements[weight_index] = weight_momentum;

            // mhat(t) = m(t) / (1 – beta1(t))
            double momentum_correction = weight_momentum / (1 - pow(beta1, simulated_training_epoch));

            // v(t) = beta2 * v(t-1) + (1 – beta2) * g(t)^2
            double old_cache = current_layer->weight_cache->data[neuron_index]->elements[weight_index];
                
            double weight_cache = beta2 * old_cache + (1 - beta2) * pow(weight_gradient, 2);
            current_layer->weight_cache->data[neuron_index]->elements[weight_index] = weight_cache;
            // vhat(t) = v(t) / (1 – beta2(t))
            double cache_correction = weight_cache / (1 - pow(beta2, simulated_training_epoch));

            // x(t) = x(t-1) – alpha * mhat(t) / (sqrt(vhat(t)) + eps)
            double old_weight = current_layer->weights->data[neuron_index]->elements[weight_index];
            double new_weight = old_weight - (learning_rate * momentum_correction) / (sqrt(cache_correction) + epsilon);

            expected_updated_weights->data[neuron_index]->elements[weight_index] = new_weight;
        }

        double  bias_gradient = current_layer->bias_gradients->elements[neuron_index];
            
        // Momentum calculations
        double bias_momentum = beta1 * current_layer->bias_momentums->elements[neuron_index] + (1 - beta1) * bias_gradient;
        current_layer->bias_momentums->elements[neuron_index] = bias_momentum;

        double momentum_correction = bias_momentum / (1 - pow(beta1, simulated_training_epoch));

        // Cache calculations
        double bias_cache = beta2 * current_layer->bias_cache->elements[neuron_index] + (1 - beta2) * pow(bias_gradient, 2);
        current_layer->bias_cache->elements[neuron_index] = bias_cache;

        double cacheCorrection = bias_cache / (1 - pow(beta2, simulated_training_epoch));


        double old_bias = current_layer->biases->elements[neuron_index];
        double new_bias = old_bias - (learning_rate * momentum_correction) / (sqrt(cacheCorrection) + epsilon);
        expected_updated_biases->elements[neuron_index] = new_bias;
    }
    
    // Call the Adam function
    mock_network->training_epoch = simulated_training_epoch;
    adam(mock_network, learning_rate, 0);

    // Compare the actual updated weights and biases with the expected accumulated values
    assert(is_equal_matrix(expected_updated_weights, mock_network->layers[0]->weights) == 1);
    assert(is_equal_vector(expected_updated_biases, mock_network->layers[0]->biases) == 1);

    free_matrix(expected_updated_weights);
    free_vector(expected_updated_biases);

    free_network(mock_network);

    log_info("Adam test passed successfully.");
}