#include "layer.h"
#include <stdlib.h>

Layer* create_layer(LayerConfig* config) {
    Layer* layer = malloc(sizeof(Layer));

    layer->num_neurons = config->num_neurons;

    // If 2 subsequent layers have X and Y neurons, then the number of weights is X*Y
    layer->weights = create_matrix(config->num_neurons, config->num_inputs);
    initialize_weights_he(config->num_inputs, config->num_neurons, layer->weights);
    
    layer->biases = create_vector(config->num_neurons);
    fill_vector_random(layer->biases, -0.5, 0.5);

    
    layer->activation_fn = config->activation_fn;

    
    layer->weight_momentums= create_matrix(layer->weights->rows, layer->weights->columns);
    layer->bias_momentums = create_vector(layer->biases->size);
    layer->weight_cache = create_matrix(layer->weights->rows, layer->weights->columns);
    layer->bias_cache = create_vector(layer->biases->size);
    
    if(config->use_regularization == 1) {
        if(config->weight_lambda != 0) {
            layer->weight_lambda = config->weight_lambda;
        }
        if(config->bias_lambda != 0) {
            layer->bias_lambda = config->bias_lambda;
        }
    }else {
        layer->weight_lambda = 0;
        layer->bias_lambda = 0;
    }
    return layer;
}

void free_layer(Layer* layer) {
    if (layer == NULL) {
        return;
    }

    // Free the resources allocated for the layer
    free_matrix(layer->weights);
    free_matrix(layer->weight_momentums);
    free_matrix(layer->weight_cache);
    
    free_vector(layer->bias_cache);
    free_vector(layer->bias_momentums);
    free_vector(layer->biases);

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
            weights->set_element(weights, i, j, rand_num);
        }
    }
}

char* serialize_layer(const Layer* layer) {
    cJSON *root = cJSON_CreateObject();
    cJSON_AddItemToObject(root, "num_neurons", cJSON_CreateNumber(layer->num_neurons));


    char* serialized_matrix = serialize_matrix(layer->weights);
    cJSON_AddItemToObject(root, "weights", cJSON_CreateRaw(serialized_matrix));

    char* serialized_vector = serialize_vector(layer->biases);
    cJSON_AddItemToObject(root, "biases", cJSON_CreateRaw(serialized_vector));
        
    // Instead of trying to serialize the function pointers, we'll serialize a name associated with the function
    char* activationFunctionName = get_activation_function_name(layer->activation_fn);
    cJSON_AddItemToObject(root, "ActivationFunction", cJSON_CreateString(activationFunctionName));
    
    char *jsonString = cJSON_PrintUnformatted(root);

    cJSON_Delete(root);

    free(serialized_matrix);
    free(serialized_vector);

    return jsonString;
}

Layer* deserialize_layer(cJSON* json) {
    Layer* layer = malloc(sizeof(Layer));

    layer->num_neurons = cJSON_GetObjectItem(json, "num_neurons")->valueint;
    layer->weights = deserialize_matrix(cJSON_GetObjectItem(json, "weights"));

    layer->biases = deserialize_vector(cJSON_GetObjectItem(json, "biases"));

    layer->activation_fn = get_activation_function_by_name(cJSON_GetObjectItem(json, "ActivationFunction")->valuestring);
    
    layer->weight_cache = NULL;
    layer->bias_cache = NULL;
    layer->weight_momentums = NULL;
    layer->bias_momentums = NULL;

    return layer;
}