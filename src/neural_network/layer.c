#include "layer.h"

Layer* create_layer(LayerConfig* config) {
    Layer* layer = malloc(sizeof(Layer));

    layer->num_neurons = config->num_neurons;

    // If 2 subsequent layers have X and Y neurons, then the number of weights is X*Y
    layer->weights = create_matrix(config->num_neurons, config->num_inputs);
    initialize_weights_he(config->num_inputs, config->num_neurons, layer->weights);
    
    layer->weight_gradients = create_matrix(layer->weights->rows, layer->weights->columns);
    fill_matrix(layer->weight_gradients, 0.0);

    layer->biases = create_vector(config->num_neurons);
    fill_vector_random(layer->biases, -0.5, 0.5);

    layer->output = create_vector(config->num_neurons);

    layer->bias_gradients = create_vector(config->num_neurons);

    layer->weighted_sums = create_vector(config->num_neurons);
    
    layer->activation_fn = config->activation_fn;

    layer->loss_wrt_wsums = create_vector(layer->num_neurons);
    
    layer->weight_momentums = create_matrix(layer->weights->rows, layer->weights->columns);
    layer->bias_momentums = create_vector(layer->biases->size);
    layer->weight_cache = create_matrix(layer->weights->rows, layer->weights->columns);
    layer->bias_cache = create_vector(layer->biases->size);
    
    if(config->use_l1_regularization == TRUE) {
        layer->l1_weight_lambda = config->l1_weight_lambda;
        layer->l1_bias_lambda = config->l1_bias_lambda;
    }else {
        layer->l1_weight_lambda = 0;
        layer->l1_bias_lambda = 0;
    }

    if(config->use_l2_regularization == TRUE) {
        layer->l2_weight_lambda = config->l2_weight_lambda;
        layer->l2_bias_lambda = config->l2_bias_lambda;
    }else {
        layer->l2_weight_lambda = 0;
        layer->l2_bias_lambda = 0;
    }
    
    return layer;
}

void free_layer(Layer* layer) {
    if (layer == NULL) {
        return;
    }

    // Free the resources allocated for the layer
    free_matrix(layer->weights);
    free_matrix(layer->weight_gradients);
    free_matrix(layer->weight_momentums);
    free_matrix(layer->weight_cache);
    
    free_vector(layer->bias_cache);
    free_vector(layer->bias_momentums);
    free_vector(layer->biases);
    free_vector(layer->bias_gradients);
    free_vector(layer->loss_wrt_wsums);
    free_vector(layer->weighted_sums);

    // Free the layer itself
    free(layer);
}

void initialize_weights_he(int inputNeuronCount, int outputNeuronCount, Matrix* weights) {
    // Calculate limit
    double limit = sqrt(2.0 / (double)inputNeuronCount);

    // Initialize weights
    for(int i = 0; i < outputNeuronCount; i++) {
        for(int j = 0; j < inputNeuronCount; j++) {
            // Generate a random number between -limit and limit
            double rand_num = (double)rand() / RAND_MAX; // This generates a random number between 0 and 1
            rand_num = rand_num * 2 * limit - limit; // This shifts the range to [-limit, limit]
            weights->data[i]->elements[j] = rand_num;
        }
    }
}

char* serialize_layer(const Layer* layer) {
    cJSON *root = cJSON_CreateObject();
    cJSON_AddItemToObject(root, "num_neurons", cJSON_CreateNumber(layer->num_neurons));
    
    char* serialized_weights = serialize_matrix(layer->weights);
    char* serialized_biases = serialize_vector(layer->biases);
    cJSON_AddItemToObject(root, "weights", cJSON_CreateRaw(serialized_weights));
    cJSON_AddItemToObject(root, "biases", cJSON_CreateRaw(serialized_biases));

    free(serialized_weights);
        
    // Instead of trying to serialize the function pointers, we'll serialize a name associated with the function
    char* activationFunctionName = get_activation_function_name(layer->activation_fn);
    cJSON_AddItemToObject(root, "ActivationFunction", cJSON_CreateString(activationFunctionName));
    
    char *jsonString = cJSON_PrintUnformatted(root);

    cJSON_Delete(root);

    return jsonString;
}

Layer* deserialize_layer(cJSON* json) {
    Layer* layer = malloc(sizeof(Layer));

    layer->num_neurons = cJSON_GetObjectItem(json, "num_neurons")->valueint;
    layer->weights = deserialize_matrix(cJSON_GetObjectItem(json, "weights"));

    layer->biases = deserialize_vector(cJSON_GetObjectItem(json, "biases"));

    layer->activation_fn = get_activation_function_by_name(cJSON_GetObjectItem(json, "ActivationFunction")->valuestring);

    layer->weighted_sums = create_vector(layer->num_neurons);

    layer->output = create_vector(layer->num_neurons);

    layer->weight_cache = NULL;
    layer->bias_cache = NULL;
    layer->weight_momentums = NULL;
    layer->loss_wrt_wsums = NULL;
    layer->bias_momentums = NULL;

    layer->weight_gradients = NULL;
    layer->bias_gradients = NULL;
    return layer;
}