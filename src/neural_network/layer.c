#include "layer.h"

Layer* create_layer(LayerConfig* config) {
    Layer* layer = malloc(sizeof(Layer));

    layer->neuronCount = config->neuronCount;

    // If 2 subsequent layers have X and Y neurons, then the number of weights is X*Y
    layer->weights = create_matrix(config->neuronCount, config->inputSize);
    initialize_weights_he(config->inputSize, config->neuronCount, layer->weights);
    
    layer->gradients = create_matrix(layer->weights->rows, layer->weights->columns);
    fill_matrix(layer->gradients, 0.0);

    layer->biases = create_vector(config->neuronCount);
    fill_vector_random(layer->biases, -0.5, 0.5);

    layer->biasGradients = create_vector(config->neuronCount);

    layer->weightedSums = create_vector(config->neuronCount);
    layer->output = create_vector(config->neuronCount);

    layer->activationFunction = config->activationFunction;

    layer->dLoss_dWeightedSums = create_vector(layer->neuronCount);
    
    layer->weightMomentums = create_matrix(layer->weights->rows, layer->weights->columns);
    layer->biasMomentums = create_vector(layer->biases->size);
    layer->weightCache = create_matrix(layer->weights->rows, layer->weights->columns);
    layer->biasCache = create_vector(layer->biases->size);
    
    if(config->shouldUseRegularization == 1) {
        if(config->weightLambda != 0) {
            layer->weightLambda = config->weightLambda;
        }
        if(config->biasLambda != 0) {
            layer->biasLambda = config->biasLambda;
        }
    }else {
        layer->weightLambda = 0;
        layer->biasLambda = 0;
    }
    return layer;
}

void free_layer(Layer* layer) {
    if (layer == NULL) {
        return;
    }

    // Free the resources allocated for the layer
    // free_matrix(layer->input);
    free_matrix(layer->weights);
    free_matrix(layer->gradients);
    free_matrix(layer->weightMomentums);
    free_matrix(layer->weightCache);
    
    free_vector(layer->biasCache);
    free_vector(layer->biasMomentums);
    free_vector(layer->biases);
    free_vector(layer->biasGradients);
    free_vector(layer->dLoss_dWeightedSums);
    free_vector(layer->output);
    free_vector(layer->weightedSums);

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
    cJSON_AddItemToObject(root, "neuronCount", cJSON_CreateNumber(layer->neuronCount));
    
    cJSON_AddItemToObject(root, "weights", cJSON_CreateRaw(serialize_matrix(layer->weights)));
    cJSON_AddItemToObject(root, "biases", cJSON_CreateRaw(serialize_vector(layer->biases)));
        
    // Instead of trying to serialize the function pointers, we'll serialize a name associated with the function
    char* activationFunctionName = get_activation_function_name(layer->activationFunction);
    cJSON_AddItemToObject(root, "activationFunction", cJSON_CreateString(activationFunctionName));
    
    char *jsonString = cJSON_PrintUnformatted(root);

    cJSON_Delete(root);

    return jsonString;
}

Layer* deserialize_layer(cJSON* json) {
    Layer* layer = malloc(sizeof(Layer));

    layer->neuronCount = cJSON_GetObjectItem(json, "neuronCount")->valueint;
    layer->weights = deserialize_matrix(cJSON_GetObjectItem(json, "weights"));
    log_debug("loaded layer's weights: %s", matrix_to_string(layer->weights));

    layer->biases = deserialize_vector(cJSON_GetObjectItem(json, "biases"));
    log_debug("loaded layer's biases: %s", vector_to_string(layer->biases));

    layer->activationFunction = malloc(sizeof(ActivationFunction));
    *layer->activationFunction = get_activation_function_by_name(cJSON_GetObjectItem(json, "activationFunction")->valuestring);

    log_debug("loaded layer's activation function: %s", get_activation_function_name(layer->activationFunction));

    return layer;
}