#include "nnetwork.h"
#include "activation_function.h"

/*
    create_network works by getting the config and the inputs.
    It allocates memory for the Neural Network based on the struct 

    typedef struct {
        Layer* start;
        Layer* end;
    } NNetwork;

    Where each layer holds information like :
    * number of neurons
    * pointer to a vector of inputs
    * Pointer to an array of Neurons
    * Pointer to a vector of outputs
    * Pointer to an activation function
    * Pointer to the next layer (Layers can classified as linked lists)
    * Pointer to the prev layer (Layers can classified as linked lists)

    
    we loop config.number of layers times and create each layer.
    we mark the first layer to be network's start layer and the last layer to be the end layer
*/
NNetwork* create_network(const NetworkConfig* config) {
    NNetwork* network = malloc(sizeof(NNetwork));
    network->layerCount = config->numLayers;
    network->layers = malloc(network->layerCount * sizeof(Layer));

   for (int i = 0; i < config->numLayers; i++) {
        LayerConfig layerConfig;
        layerConfig.inputSize = i == 0 ? config->num_features : network->layers[i - 1]->neuronCount;
        layerConfig.neuronCount = config->neuronsPerLayer[i];
        layerConfig.activationFunction = &config->activationFunctions[i];

        if(i < config->numLayers - 1){
            int use_regularization = 0;

            if(config->weightLambdas != NULL && config->weightLambdas->size > 0){
                use_regularization = 1;
                layerConfig.weightLambda = config->weightLambdas->elements[i]; 
            }else {
                layerConfig.weightLambda = 0;
            }

            if(config->biasLambdas != NULL && config->biasLambdas->size > 0) {
                use_regularization = 1;
                layerConfig.biasLambda = config->biasLambdas->elements[i];
            }else {
                layerConfig.biasLambda = 0;
            }

            log_debug("should use regularization? %d", use_regularization);
            layerConfig.shouldUseRegularization = use_regularization;
        }
        Layer* layer = create_layer(&layerConfig);
        network->layers[i] = layer;
    }
    

    network->lossFunction = config->lossFunction;

    network->optimizationConfig = config->optimizationConfig;

    network->loss = 0.0f;
    network->accuracy = 0.0f;
    
    switch (network->optimizationConfig->optimizer) {
        case SGD:
            network->optimizer = sgd;
            break;
        case ADAGRAD:
            network->optimizer = adagrad;
            break;
        case RMS_PROP:
            network->optimizer = rms_prop;
            break;
        case ADAM:
            network->optimizer = adam;
            break;
        default:
            break;
    }
    network->output = create_matrix_arr(network->layerCount);
    network->weightedsums = create_matrix_arr(network->layerCount);
    network->weight_gradients = create_matrix_arr(network->layerCount);
    network->bias_gradients = create_vector_arr(network->layerCount);

    for(int i = 0; i < network->layerCount; i++) {
        // I allocate memory for vectors using calloc so they are already initialized to zeroes!
        network->output[i] = create_matrix(config->num_rows, network->layers[i]->neuronCount);
        network->weightedsums[i] = create_matrix(config->num_rows, network->layers[i]->neuronCount);
        network->weight_gradients[i] = create_matrix(network->layers[i]->weights->rows, network->layers[i]->weights->columns);
        network->bias_gradients[i] = create_vector(network->layers[i]->biases->size);
    }

    log_info("%s", "Created Network:");
    dump_network_config(network);

    return network;
}
/*
    This method performs the forward pass of a neural network using a batched processing approach. 
    It processes the entire input batch in a vectorized manner, taking advantage of parallelism and optimized matrix operations.

    Use this only if your computer can support multithreading and/or CUDA.
*/
void forward_pass_batched(NNetwork* network, Matrix* input_matrix) { 
    for(int i = 0; i < network->layerCount; i++) {

        Layer* current_layer = network->layers[i];
        Matrix* transposed_weights = matrix_transpose(current_layer->weights);
        
        // do not free this, it points to the original input matrix for the first layer!
        Matrix* layer_input = i == 0 ? input_matrix : network->output[i - 1];

        Matrix* product_result = matrix_product(layer_input, transposed_weights);
        // log_info("product result for first layer: %s", matrix_to_string(product));
        
        matrix_vector_addition(product_result, current_layer->biases, network->weightedsums[i]);
        // log_info("weighted sums: %s", matrix_to_string(network->weightedsums[i]));


        copy_matrix_into(network->weightedsums[i], network->output[i]);
        // log_info("output: %s", matrix_to_string(network->output[i]));

        if(i == network->layerCount - 1) {
            softmax_matrix(network->output[i]);
        }else {
            leakyRelu(network->output[i]->data);
        }

        free_matrix(product_result);
        free_matrix(transposed_weights);
    }
}

void calculate_loss(NNetwork* network, Matrix* yValues) {
    network->loss = categoricalCrossEntropyLoss(yValues, network->output[network->layerCount - 1]);

    network->accuracy = accuracy(yValues, network->output[network->layerCount - 1]);
}

void backpropagation_batched(NNetwork* network, Matrix* input_matrix, Matrix* y_values) {
    #ifdef DEBUG
        char* debug_str;
    #endif
    int num_layers = network->layerCount;
    Matrix** loss_wrt_weightedsum = create_matrix_arr(num_layers);

    // -------------OUTPUT LAYER-------------
    int layer_index = num_layers - 1;
    Matrix* output = network->output[layer_index];

    // I can distribute the work amongst the threads in the thread pool for all three operations.
    Matrix* loss_wrt_output = create_matrix(y_values->rows, y_values->columns);
    computeCategoricalCrossEntropyLossDerivativeMatrix(y_values, output, loss_wrt_output);
    #ifdef DEBUG
        debug_str = matrix_to_string(loss_wrt_output);
        log_info("loss_wrt_output: %s", debug_str);
        free(debug_str);
    #endif

    Matrix** jacobian_matrices = softmax_derivative_parallelized(output);

    loss_wrt_weightedsum[layer_index] = matrix_vector_product_arr(jacobian_matrices, loss_wrt_output, output->rows);
    #ifdef DEBUG
        debug_str = matrix_to_string(loss_wrt_weightedsum[layer_index]);
        log_info("Loss wrt WeightedSum matrix for layer #%d: %s", layer_index, debug_str);
        free(debug_str);
    #endif

    // clean memory
    free_matrix(loss_wrt_output);

    for(int i = 0; i < output->rows; i++) {
        free_matrix(jacobian_matrices[i]);
    }
    free(jacobian_matrices);

    Matrix* weightedsum_wrt_weight = NULL;    

    if(layer_index == 0) {
        weightedsum_wrt_weight = input_matrix;
    }else {
        weightedsum_wrt_weight = network->output[layer_index - 1];
    }
    
    #ifdef DEBUG
        debug_str = matrix_to_string(weightedsum_wrt_weight);
        log_info("weightedsum wrt weights for layer #%d: %s", layer_index, debug_str);
        free(debug_str);
    #endif

    // multiplying each weighted sum with different input neurons to get the gradients of the weights that connect them
    for(int input_index = 0; input_index < weightedsum_wrt_weight->rows; input_index++) {
        for(int i = 0; i < loss_wrt_weightedsum[layer_index]->columns; i++) {
            for(int j = 0; j < weightedsum_wrt_weight->columns; j++) {
                
                double product_result = loss_wrt_weightedsum[layer_index]->get_element(loss_wrt_weightedsum[layer_index], input_index, i) 
                                        * weightedsum_wrt_weight->get_element(weightedsum_wrt_weight, input_index, j);

                double value = network->weight_gradients[layer_index]->get_element(network->weight_gradients[layer_index], i, j) + product_result;
                network->weight_gradients[layer_index]->set_element(network->weight_gradients[layer_index], i, j, value);
                // network->weight_gradients[layer_index]->data[i]->elements[j] += loss_wrt_weightedsum[layer_index]->data[input_index]->elements[i] * weightedsum_wrt_weight->data[input_index]->elements[j];
            }
        }
    }

    #ifdef DEBUG
        debug_str = matrix_to_string(network->weight_gradients[layer_index]);
        log_info("Weight gradients of the output layer: %s", debug_str);
        free(debug_str);
    #endif

    // if(useGradientClipping) clip_gradients(weight_gradients)

    for(int i = 0; i < loss_wrt_weightedsum[layer_index]->rows; i++) {
        for(int j = 0; j < loss_wrt_weightedsum[layer_index]->columns; j++) {
            network->bias_gradients[layer_index]->elements[j] += loss_wrt_weightedsum[layer_index]->get_element(loss_wrt_weightedsum[layer_index], i, j);
        }
    }
    #ifdef DEBUG
        debug_str = vector_to_string(network->bias_gradients[layer_index]);
        log_info("Bias gradients for the output layer: %s", debug_str);
        free(debug_str);
    #endif
    // if(useGradientClipping) clip_gradients(bias_gradients)

    
    // ------------- HIDDEN LAYERS -------------
    // we do need to iterate over other layers
    for (layer_index -= 1; layer_index >= 0; layer_index--) { // current layer's dimensions = (4 inputs, 4 neurons)

        Matrix* loss_wrt_output = matrix_product(loss_wrt_weightedsum[layer_index + 1], network->layers[layer_index + 1]->weights);
        #ifdef DEBUG
            debug_str = matrix_to_string(loss_wrt_output);
            log_info("loss wrt output for layer: #%d: %s", layer_index, debug_str);
            free(debug_str);
        #endif


        Matrix* output_wrt_weightedsums = leakyRelu_derivative_matrix(network->weightedsums[layer_index]);
        #ifdef DEBUG
            char* debug_str = matrix_to_string(output_wrt_weightedsums);
            log_info("output wrt wsum for layer #%d: %s", layer_index, debug_str);
            free(debug_str);
        #endif
        
        matrix_multiplication(loss_wrt_output, output_wrt_weightedsums, loss_wrt_weightedsum[layer_index]);
        
        free_matrix(loss_wrt_output);

        free_matrix(output_wrt_weightedsums);
        free(output_wrt_weightedsums);
        
        #ifdef DEBUG
            debug_str = matrix_to_string(loss_wrt_weightedsum[layer_index]);
            log_info("loss wrt weighted sum for layer #%d: %s", layer_index, debug_str);
            free(debug_str);
        #endif

        if (layer_index == 0) {
            weightedsum_wrt_weight = input_matrix;
        } else {
            weightedsum_wrt_weight = network->output[layer_index - 1];
        }

        // log_info("weightedsum wrt weights for layer #%d: %s", layer_index, matrix_to_string(weightedsum_wrt_weight));
        
        // multiplying each weighted sum with different input neurons to get the gradients of the weights that connect them
        for(int input_index = 0; input_index < weightedsum_wrt_weight->rows; input_index++) {
            for(int i = 0; i < loss_wrt_weightedsum[layer_index]->columns; i++) {
                for(int j = 0; j < weightedsum_wrt_weight->columns; j++) {
                    double product_result = loss_wrt_weightedsum[layer_index]->get_element(loss_wrt_weightedsum[layer_index], input_index, i) 
                                        * weightedsum_wrt_weight->get_element(weightedsum_wrt_weight, input_index, j);

                    double value = network->weight_gradients[layer_index]->get_element(network->weight_gradients[layer_index], i, j) + product_result;
                    network->weight_gradients[layer_index]->set_element(network->weight_gradients[layer_index], i, j, value);
                    // network->weight_gradients[layer_index]->data[i]->elements[j] += loss_wrt_weightedsum[layer_index]->data[input_index]->elements[i] * input_matrix->data[input_index]->elements[j];
                }
            }
        }
        #ifdef DEBUG
            debug_str = matrix_to_string(network->weight_gradients[layer_index]);
            log_info("Weight gradients of the layer #%d: %s", layer_index, debug_str);
            free(debug_str);
        #endif
        // if(useGradientClipping) clip_gradients(weight_gradients)

        for(int i = 0; i < loss_wrt_weightedsum[layer_index]->rows; i++) {
            for(int j = 0; j < loss_wrt_weightedsum[layer_index]->columns; j++) {
                network->bias_gradients[layer_index]->elements[j] += loss_wrt_weightedsum[layer_index]->get_element(loss_wrt_weightedsum[layer_index], i, j);
            }
        }
        #ifdef DEBUG
            debug_str = vector_to_string(network->bias_gradients[layer_index]);
            log_info("Bias gradients of the layer #%d: %s", layer_index, debug_str);
            free(debug_str);
        #endif
    }

    for(int i = 0; i < num_layers; i++) {
        free_matrix(loss_wrt_weightedsum[i]);
        free(loss_wrt_weightedsum[i]);
    }

    free(loss_wrt_weightedsum);
} 

double accuracy(Matrix* targets, Matrix* outputs) {
    int counter = 0;
    // todo: change this back to output.rows
    for(int i = 0; i < outputs->rows; i++){
        int maxIndexOutputs = arg_max_matrix_row(outputs, i);
        int maxIndexTargets = arg_max_matrix_row(targets, i);
   
        if (maxIndexOutputs == maxIndexTargets) {
            counter++;
        } 

        #ifdef DEBUG
            char* targetVectorStr = vector_to_string(targets->data[i]);
            char* outputVectorStr = vector_to_string(outputs->data[i]);
            log_debug(
                "Accuracy Calculation: \n"
                "\t\t\t\tTarget Vector: %s \n"
                "\t\t\t\tOutput Vector: %s \n"
                "\t\t\t\tPredicted Category: %d \n"
                "\t\t\t\tTarget Category: %d \n"
                "\t\t\t\tCorrect Predictions Count: %d \n", 
                targetVectorStr, 
                outputVectorStr, 
                maxIndexOutputs, 
                maxIndexTargets, 
                counter
            );
            free(targetVectorStr);
            free(outputVectorStr);
        #endif

    }
    // todo: change this back to output.rows
    double averageAccuracy = (double)counter / outputs->rows;

    #ifdef DEBUG
        log_debug("Average Accuracy: %f \n", averageAccuracy);
    #endif

    return averageAccuracy;
}

void dump_network_config(NNetwork* network) {
   char json_output[5000]; // adjust the size to suit your needs

    sprintf(
        json_output,
        "{\n"
        "\t\"Network Config\": {\n"
        "\t\t\"Loss Function\": %s \n"
        "\t\t\"Layers\": [\n",
        "cross_entropy" // TODO: Turn into const values
    );

    for(int i = 0; i < network->layerCount; i++) {
        Layer* layer = network->layers[i];
        sprintf(
            json_output + strlen(json_output), // add to the end of the current string
            "\t\t\t{\n"
            "\t\t\t\t\"Layer Index\": %d,\n"
            "\t\t\t\t\"Number Of Neurons\": %d,\n"
            "\t\t\t\t\"Input Size\": %d,\n"
            "\t\t\t\t\"Weight Initialization Method\": \"%s\",\n"
            "\t\t\t\t\"Biases in Range\": [%f, %f]\n"
            "\t\t\t\t\"Activation Function\": %s \n" // TODO: Actually read this from the layer struct
            "\t\t\t}\n%s", // add comma for all but last layer
            i, layer->neuronCount, layer->weights->columns, "he initializer", -.5, .5, 
            i < network->layerCount - 1 ? "leaky relu" : "softmax",
            (i == network->layerCount - 1) ? "" : ",\n"
        );
    }

    sprintf(json_output + strlen(json_output), // add to the end of the current string
        "\t\t\t]\n"
        "\t}\n"
    );

    OptimizationConfig* opt_config = network->optimizationConfig/* get your OptimizationConfig object here */;

    sprintf(
        json_output + strlen(json_output),
        "\t\t\"Optimizer Config\": {\n"
        "\t\t\t\"shouldUseGradientClipping\": %d,\n"
        "\t\t\t\"gradientClippingLowerBound\": %f,\n"
        "\t\t\t\"gradientClippingUpperBound\": %f,\n"
        "\t\t\t\"shouldUseLearningRateDecay\": %d,\n"
        "\t\t\t\"learningRateDecayAmount\": %f,\n"
        "\t\t\t\"shouldUseMomentum\": %d,\n"
        "\t\t\t\"momentum\": %f,\n"
        "\t\t\t\"optimizer\": %s,\n"
        "\t\t\t\"epsilon\": %f,\n"
        "\t\t\t\"rho\": %f,\n"
        "\t\t\t\"beta1\": %f,\n"
        "\t\t\t\"beta2\": %f\n"
        "\t\t}\n", // add a comma as this might not be the last item in its parent object
        opt_config->shouldUseGradientClipping,
        opt_config->shouldUseGradientClipping == 1 ? opt_config->gradientClippingLowerBound : 0,
        opt_config->shouldUseGradientClipping == 1 ? opt_config->gradientClippingUpperBound : 0,
        opt_config->shouldUseLearningRateDecay,
        opt_config->learningRateDecayAmount,
        opt_config->shouldUseMomentum,
        opt_config->momentum,
        get_optimizer_name(opt_config->optimizer),
        opt_config->epsilon,
        opt_config->rho,
        opt_config->beta1,
        opt_config->beta2
    );

    log_info("%s", json_output);

}

double calculate_l1_penalty(double lambda, const Vector* vector) {
    double penalty = 0.0f;

    for(int i = 0; i < vector->size; i++) {
        penalty += fabs(vector->elements[i]);
    }

    return lambda * penalty;
}

double calculate_l2_penalty(double lambda, const Vector* vector) {
    double penalty = 0.0f;

    for(int i = 0; i < vector->size; i++) {
        penalty += pow(vector->elements[i], 2);
    }

    return lambda * penalty;
}

Vector* l1_derivative(double lambda, const Vector* vector) {
    Vector* derivatives = create_vector(vector->size);

    for(int i = 0; i < derivatives->size; i++) {
        if(vector->elements[i] == 0) {
            derivatives->elements[i] = 0;
        }else if(vector->elements[i] > 0) {
            derivatives->elements[i] = lambda * 1;
        }else{
            derivatives->elements[i] = lambda * -1;
        }
    }

    return derivatives;
}

/*
    The L2 regularization penalty is lambda * sum(weights ^ 2).

    To get the derivative, we take the derivative of that penalty term with 
    respect to each weight parameter w:

    d/dw (lambda * w^2) = 2 * lambda * w

    So the contribution of each weight w to the total gradient is proportional 
    to 2 * lambda * w.
*/
Vector* l2_derivative(double lambda, const Vector* vector) {
    Vector* derivatives = create_vector(vector->size);

    for(int i = 0; i < derivatives->size; i++) {
        derivatives->elements[i] = 2 * lambda * vector->elements[i];
    }

    return derivatives;
}

void free_network(NNetwork* network) {
    if (network == NULL) {
        return;
    }

    // Free the layers
    for (int i = 0; i < network->layerCount; i++) {
        free_layer(network->layers[i]);

        free_matrix(network->weight_gradients[i]);
        
        free_vector(network->bias_gradients[i]);

        free_matrix(network->weightedsums[i]);

        free_matrix(network->output[i]);
    }

    free(network->weightedsums);
    free(network->output);
    free(network->layers);
    free(network->weight_gradients);
    free(network->bias_gradients);
    
    // Free the loss function
    free(network->lossFunction);
    free(network->optimizationConfig);

    // Finally, free the network itself
    free(network);
}

void free_network_config(NetworkConfig* config) { 
    free(config->neuronsPerLayer);

    free_vector(config->weightLambdas);
    free_vector(config->biasLambdas);
}

char* serialize_optimization_config(OptimizationConfig* config) {
     if (config == NULL) {
        printf("Config is NULL\n");
        return NULL;
    }

    if(config->shouldUseGradientClipping == 0) {
        config->gradientClippingLowerBound = 0.0f;
        config->gradientClippingUpperBound = 0.0f;
    }

    if(config->optimizer != SGD){
        config->shouldUseMomentum = 0;
        config->momentum = 0.0f;
    }
    
    if(config->optimizer != RMS_PROP) {
        config->rho = 0.0f;
    }

    if(config->optimizer != ADAM) {
        config->beta1 = 0.0f;
        config->beta2 = 0.0f;
    }
    cJSON *root = cJSON_CreateObject();

    cJSON_AddItemToObject(root, "shouldUseGradientClipping", cJSON_CreateNumber(config->shouldUseGradientClipping));
    
    cJSON_AddItemToObject(root, "gradientClippingLowerBound", cJSON_CreateNumber(config->gradientClippingLowerBound));
    cJSON_AddItemToObject(root, "gradientClippingUpperBound", cJSON_CreateNumber(config->gradientClippingUpperBound));
    cJSON_AddItemToObject(root, "shouldUseLearningRateDecay", cJSON_CreateNumber(config->shouldUseLearningRateDecay));
    cJSON_AddItemToObject(root, "learningRateDecayAmount", cJSON_CreateNumber(config->learningRateDecayAmount));
    cJSON_AddItemToObject(root, "shouldUseMomentum", cJSON_CreateNumber(config->shouldUseMomentum));
    cJSON_AddItemToObject(root, "momentum", cJSON_CreateNumber(config->momentum));
    cJSON_AddItemToObject(root, "optimizer", cJSON_CreateNumber(config->optimizer));
    cJSON_AddItemToObject(root, "epsilon", cJSON_CreateNumber(config->epsilon));
    cJSON_AddItemToObject(root, "rho", cJSON_CreateNumber(config->rho));
    cJSON_AddItemToObject(root, "beta1", cJSON_CreateNumber(config->beta1));
    cJSON_AddItemToObject(root, "beta2", cJSON_CreateNumber(config->beta2));

    char *jsonString = cJSON_PrintUnformatted(root);

    cJSON_Delete(root);

    return jsonString;
}

char* serialize_network(const NNetwork* network) {
    cJSON *root = cJSON_CreateObject();
    cJSON *layers = cJSON_CreateArray();

    for (int i = 0; i < network->layerCount; i++) {
        char *layerString = serialize_layer(network->layers[i]);
        cJSON_AddItemToArray(layers, cJSON_CreateRaw(layerString));
        free(layerString);
    }

    cJSON_AddItemToObject(root, "layerCount", cJSON_CreateNumber(network->layerCount));
    cJSON_AddItemToObject(root, "layers", layers);
    cJSON_AddItemToObject(root, "lossFunction", cJSON_CreateString(get_loss_function_name(network->lossFunction)));
    cJSON_AddItemToObject(root, "loss", cJSON_CreateNumber(network->loss));
    cJSON_AddItemToObject(root, "accuracy", cJSON_CreateNumber(network->accuracy));

    char *jsonString = cJSON_Print(root);
    cJSON_Delete(root);
   
    return jsonString;
}

NNetwork* deserialize_network(cJSON* json) {
    NNetwork* network = malloc(sizeof(NNetwork));
    if (network == NULL) {
        return NULL;
    }

    network->layerCount = cJSON_GetObjectItem(json, "layerCount")->valueint;
    network->layers = malloc(network->layerCount * sizeof(Layer));
    network->weightedsums = create_matrix_arr(network->layerCount);

    network->weight_gradients = create_matrix_arr(network->layerCount);
    
    network->bias_gradients = create_vector_arr(network->layerCount);

    network->output = create_matrix_arr(network->layerCount);

    cJSON* json_layers = cJSON_GetObjectItem(json, "layers");
   
    for (int i = 0; i < network->layerCount; i++) {
        cJSON* json_layer = cJSON_GetArrayItem(json_layers, i);
        network->layers[i] = deserialize_layer(json_layer);
    }

    network->lossFunction = NULL;
    network->optimizationConfig = NULL;

    // network->lossFunction = strdup(cJSON_GetObjectItem(json, "lossFunction")->valuestring);
    // network->loss = cJSON_GetObjectItem(json, "loss")->valuedouble;
    // network->accuracy = cJSON_GetObjectItem(json, "accuracy")->valuedouble;
        
    return network;
}

int save_network(char* path, NNetwork* network) {
    // Serialize the network
    char *networkJson = serialize_network(network);
    if (networkJson == NULL) {
        return -1;
    }

    // Check if the path has a .json extension, and add it if not
    char *jsonPath;
    if (strstr(path, ".json") != NULL) {
        jsonPath = path;
    } else {
        jsonPath = malloc(strlen(path) + 6);  // Extra space for ".json" and null terminator
        if (jsonPath == NULL) {
            free(networkJson);
            return -1;
        }
        sprintf(jsonPath, "%s.json", path);
    }

    // Open the file
    FILE *file = fopen(jsonPath, "w");
    if (file == NULL) {
        free(networkJson);
        if (jsonPath != path) {
            free(jsonPath);
        }
        return -1;
    }

    // Write the serialized network to the file
    fprintf(file, "%s", networkJson);

    // Clean up
    free(networkJson);
    fclose(file);
    if (jsonPath != path) {
        free(jsonPath);
    }

    return 0;
}

NNetwork* load_network(char* path) {
    // Check if the path has a .json extension, and add it if not
    char *jsonPath;
    if (strstr(path, ".json") != NULL) {
        jsonPath = path;
    } else {
        jsonPath = malloc(strlen(path) + 6);  // Extra space for ".json" and null terminator
        if (jsonPath == NULL) {
            return NULL;
        }
        sprintf(jsonPath, "%s.json", path);
    }

    // Open the file
    FILE *file = fopen(jsonPath, "r");
    if (file == NULL) {
        if (jsonPath != path) {
            free(jsonPath);
        }
        return NULL;
    }

    // Get the size of the file
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Read the file into a string
    char *json = malloc(length + 1);
    if (json == NULL) {
        fclose(file);
        if (jsonPath != path) {
            free(jsonPath);
        }
        return NULL;
    }
    fread(json, 1, length, file);
    json[length] = '\0';

    // Close the file
    fclose(file);

    // Parse the string into a cJSON object
    cJSON *jsonObject = cJSON_Parse(json);
    if (jsonObject == NULL) {
        free(json);
        if (jsonPath != path) {
            free(jsonPath);
        }
        return NULL;
    }

    // Convert the cJSON object into a NNetwork object
    NNetwork *network = deserialize_network(jsonObject);  // Assuming this function exists

    // Clean up
    cJSON_Delete(jsonObject);
    free(json);
    if (jsonPath != path) {
        free(jsonPath);
    }

    return network;
}