#include "nnetwork.h"

NNetwork* create_network(const NetworkConfig* config) {
    NNetwork* network = malloc(sizeof(NNetwork));
    network->num_layers = config->num_layers;
    network->layers = malloc(network->num_layers * sizeof(Layer));
    network->optimization_config = config->optimization_config;

   for (int layer_index = 0; layer_index < config->num_layers; layer_index++) {
        LayerConfig layer_config;
        layer_config.num_inputs = layer_index == 0 ? config->num_features : network->layers[layer_index - 1]->num_neurons;
        layer_config.num_neurons = config->neurons_per_layer[layer_index];
        layer_config.activation_fn = config->activation_fns[layer_index];


        if(layer_index < config->num_layers - 1){
            int use_regularization = 0;

            if(network->optimization_config->use_l1_regularization == TRUE) {
                layer_config.use_l1_regularization = TRUE;
                layer_config.l1_weight_lambda = network->optimization_config->l1_weight_lambdas->elements[layer_index];
                layer_config.l1_bias_lambda = network->optimization_config->l1_bias_lambdas->elements[layer_index];
            }

            if(network->optimization_config->use_l2_regularization == TRUE) {
                layer_config.use_l2_regularization = TRUE;
                layer_config.l2_weight_lambda = network->optimization_config->l2_weight_lambdas->elements[layer_index];
                layer_config.l2_bias_lambda = network->optimization_config->l2_bias_lambdas->elements[layer_index];
            }
        }
        Layer* layer = create_layer(&layer_config);
        network->layers[layer_index] = layer;
    }
    

    network->loss_fn = config->loss_fn;


    network->loss = 0.0f;
    network->accuracy = 0.0f;
    
    switch (network->optimization_config->optimizer) {
        case SGD:
            network->optimization_algorithm = sgd;
            break;
        case ADAGRAD:
            network->optimization_algorithm = adagrad;
            break;
        case RMS_PROP:
            network->optimization_algorithm = rms_prop;
            break;
        case ADAM:
            network->optimization_algorithm = adam;
            break;
        default:
            break;
    }
    network->batched_outputs = create_matrix_arr(network->num_layers);

    network->weighted_sums = create_matrix_arr(network->num_layers);

    network->weight_gradients = create_matrix_arr(network->num_layers);
    for(int i = 0; i < network->num_layers; i++) {
        network->weight_gradients[i] = create_matrix(network->layers[i]->weights->rows, network->layers[i]->weights->columns);
        fill_matrix(network->weight_gradients[i], 0.0f);
    }

    network->bias_gradients = create_vector_arr(network->num_layers);
    for(int i = 0; i < network->num_layers; i++) {
        network->bias_gradients[i] = create_vector(network->layers[i]->biases->size);
        fill_vector(network->bias_gradients[i], 0.0f);
    }

    return network;
}

void train_network(NNetwork* network, Matrix* training_data, Matrix* training_labels, int batch_size, int num_epochs, double learning_rate) {
    
    int epoch = 1; // We start from 1 because having the epoch be 0 messes up the optimization calculation.

    // default rate of keras -> 0.001
    // kaparthy's recommendation for adam: 0.0003
    double current_learning_rate = learning_rate;

    double min_loss = __DBL_MAX__;
    double max_accuracy = 0.0;
    if(batch_size == 0) {
        network->output = create_matrix(training_data->rows, network->layers[network->num_layers - 1]->num_neurons);
    }

    while(epoch < num_epochs) {
        if(batch_size == 1) {
            forward_pass_batched(network, training_data); 
            backpropagation_batched(network, training_data, training_labels);

            calculate_loss(network, training_labels, network->batched_outputs[network->num_layers - 1]);
        }else if(batch_size == 0) {
            for(int inputRow = 0; inputRow < training_data->rows; inputRow++) {
                Vector* output = create_vector(network->layers[network->num_layers - 1]->num_neurons);
                forward_pass_sequential(network, training_data->data[inputRow], output); 
                backpropagation_sequential(network, training_data->data[inputRow], output, training_labels->data[inputRow]);
                network->output->data[inputRow] = copy_vector(output);
                free_vector(output);
            }

            calculate_loss(network, training_labels, network->output);
        }else {
            log_error("Haven't implemented multiple batch processing yet!");
            return;
        }

        if(network->optimization_config->use_learning_rate_decay == 1) {
            double decayRate = network->optimization_config->learning_rate_decay_amount;
            current_learning_rate = current_learning_rate * (1 / (1.0 + (decayRate * (double)epoch)));
        }

        network->training_epoch = epoch;
        network->optimization_algorithm(network, current_learning_rate, batch_size);
        
        if(epoch == 1 || epoch % 10 == 0){
            log_debug("Epoch: %d, Accuracy: %f, Loss: %f \n", epoch, network->accuracy, network->loss);  
        }

        min_loss = fmin(min_loss, network->loss);
        
        max_accuracy = fmax(max_accuracy, network->accuracy);

        epoch++;

        for(int layerIndex = 0; layerIndex < network->num_layers; layerIndex++) {
            if(batch_size == 1) {
                fill_matrix(network->weight_gradients[layerIndex], 0.0f);
                fill_vector(network->bias_gradients[layerIndex], 0.0f);
            }else if(batch_size == 0) {
                fill_matrix(network->layers[layerIndex]->weight_gradients, 0.0f);
                fill_vector(network->layers[layerIndex]->bias_gradients, 0.0f);
            }
        }
    }

    log_info("Minimum loss during training: %f \n", min_loss);
    log_info("Maximum accuracy during training: %f \n", max_accuracy);    
}

void forward_pass_batched(NNetwork* network, Matrix* input_matrix) { 
    for(int layer_index = 0; layer_index < network->num_layers; layer_index++) {
        Matrix* product_result = NULL;

        Layer* current_layer = network->layers[layer_index];
        Matrix* transposed_weights = matrix_transpose(current_layer->weights);
        
        if(layer_index == 0) {
            product_result = matrix_product(input_matrix, transposed_weights);
            // log_info("product result for first layer: %s", matrix_to_string(product));
        }else{
            product_result = matrix_product(network->batched_outputs[layer_index - 1], transposed_weights);
            // log_info("product result for layer #%d: %s", layer_index, matrix_to_string(product));
        }
        
        network->weighted_sums[layer_index] = matrix_vector_addition(product_result, current_layer->biases);
        
        network->batched_outputs[layer_index] = copy_matrix(network->weighted_sums[layer_index]);

        switch(current_layer->activation_fn) {
            case LEAKY_RELU:
                leakyReluMatrix(network->batched_outputs[layer_index]);
                break;
            case SOFTMAX:
                softmax_matrix(network->batched_outputs[layer_index]);
                break;
            default:
                log_error("Unknown Activation Function be sure to register it to the workflow.", get_activation_function_name(current_layer->activation_fn));
                break;
        }
        free_matrix(transposed_weights);

        free_matrix(product_result);
        free(product_result);
    }
}


void calculate_loss(NNetwork* network, Matrix* target_values, Matrix* output) {
    switch(network->loss_fn) {
        case MEAN_SQUARED_ERROR:            
            network->loss = mean_squared_error(output, target_values);
            break;
        case CATEGORICAL_CROSS_ENTROPY:
            network->loss = categorical_cross_entropy_loss(target_values, output);
            network->accuracy = accuracy(target_values, output);
            break;
        default:
            log_error("Unrecognized Loss Function, Please register your loss function!");
            return;
    }
}


void backpropagation_batched(NNetwork* network, Matrix* input_matrix, Matrix* y_values) {
    #ifdef DEBUG
        char* debug_str;
    #endif
    int num_layers = network->num_layers;
    Matrix** loss_wrt_weightedsum = create_matrix_arr(num_layers);

    // -------------OUTPUT LAYER-------------
    int layer_index = num_layers - 1;
    Matrix* output = network->batched_outputs[layer_index];

    Matrix* loss_wrt_output = create_matrix(y_values->rows, y_values->columns);
    categorical_cross_entropy_loss_derivative_batched(y_values, output, loss_wrt_output);
    #ifdef DEBUG
        debug_str = matrix_to_string(loss_wrt_output);
        log_info("loss_wrt_output: %s", debug_str);
        free(debug_str);
    #endif

    // Also known as jacobian matrices
    Matrix** output_wrt_weightedsum = softmax_derivative_parallelized(output);

    loss_wrt_weightedsum[layer_index] = matrix_vector_product_arr(output_wrt_weightedsum, loss_wrt_output, output->rows);
    #ifdef DEBUG
        debug_str = matrix_to_string(loss_wrt_weightedsum[layer_index]);
        log_info("Loss wrt WeightedSum matrix for layer #%d: %s", layer_index, debug_str);
        free(debug_str);
    #endif

    Matrix* weightedsum_wrt_weight = network->batched_outputs[layer_index - 1];    

    #ifdef DEBUG
        debug_str = matrix_to_string(weightedsum_wrt_weight);
        log_info("weightedsum wrt weights for layer #%d: %s", layer_index, debug_str);
        free(debug_str);
    #endif

    // multiplying each weighted sum with different input neurons to get the gradients of the weights that connect them
    calculate_weight_gradients(network, layer_index, loss_wrt_weightedsum[layer_index], weightedsum_wrt_weight);

    #ifdef DEBUG
        debug_str = matrix_to_string(network->weight_gradients[layer_index]);
        log_info("Weight gradients of the output layer: %s", debug_str);
        free(debug_str);
    #endif

    // if(useGradientClipping) clip_gradients(weight_gradients)

    for(int i = 0; i < loss_wrt_weightedsum[layer_index]->rows; i++) {
        for(int j = 0; j < loss_wrt_weightedsum[layer_index]->columns; j++) {
            network->bias_gradients[layer_index]->elements[j] += loss_wrt_weightedsum[layer_index]->data[i]->elements[j];
        }
    }
    #ifdef DEBUG
        debug_str = vector_to_string(network->bias_gradients[layer_index]);
        log_info("Bias gradients for the output layer: %s", debug_str);
        free(debug_str);
    #endif
    // if(useGradientClipping) clip_gradients(bias_gradients)

    // clean memory
    free_matrix(loss_wrt_output);

    for(int i = 0; i < output->rows; i++) {
        free_matrix(output_wrt_weightedsum[i]);
    }
    free(output_wrt_weightedsum);
    
    // ------------- HIDDEN LAYERS -------------
    for (layer_index -= 1; layer_index >= 0; layer_index--) {
        // for each input row, we store loss_wrt_output of neurons in the columns
        // each column in each row will be summation of all of the next layer's loss_wrt_weighted sum values and next layer's weights
        Matrix* loss_wrt_output = matrix_product(loss_wrt_weightedsum[layer_index + 1], network->layers[layer_index + 1]->weights);
        #ifdef DEBUG
            debug_str = matrix_to_string(loss_wrt_output);
            log_info("loss wrt output for layer: #%d: %s", layer_index, debug_str);
            free(debug_str);
        #endif

        Matrix* output_wrt_weightedsums = NULL;
        switch(network->layers[layer_index]->activation_fn) {
            case RELU:
                log_info("not implemented yet!");
                return;
            case LEAKY_RELU:
                output_wrt_weightedsums = leakyRelu_derivative_matrix(network->weighted_sums[layer_index]);
                break;
            case SOFTMAX:
                log_error("cannot/shouldn't be softmax in the hidden layers.");
                return;
            case UNRECOGNIZED_AFN:
                log_error("Unrecognized activation function!");
                return;
        }

        #ifdef DEBUG
            char* debug_str = matrix_to_string(output_wrt_weightedsums);
            log_info("output wrt wsum for layer #%d: %s", layer_index, debug_str);
            free(debug_str);
        #endif
        
        loss_wrt_weightedsum[layer_index] = matrix_multiplication(loss_wrt_output, output_wrt_weightedsums);
        
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
            weightedsum_wrt_weight = network->batched_outputs[layer_index - 1];
        }
        
        calculate_weight_gradients(network, layer_index, loss_wrt_weightedsum[layer_index], weightedsum_wrt_weight);
        #ifdef DEBUG
            debug_str = matrix_to_string(network->weight_gradients[layer_index]);
            log_info("Weight gradients of the layer #%d: %s", layer_index, debug_str);
            free(debug_str);
        #endif
        
        // if(useGradientClipping) clip_gradients(weight_gradients)

        for(int i = 0; i < loss_wrt_weightedsum[layer_index]->rows; i++) {
            for(int j = 0; j < loss_wrt_weightedsum[layer_index]->columns; j++) {
                network->bias_gradients[layer_index]->elements[j] += loss_wrt_weightedsum[layer_index]->data[i]->elements[j];
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

void calculate_weight_gradients(NNetwork* network, int layer_index, Matrix* loss_wrt_weightedsum, Matrix* wsum_wrt_weight) {    
     // multiplying each weighted sum with different input neurons to get the gradients of the weights that connect them
    for (int input_index = 0; input_index < wsum_wrt_weight->rows; input_index++) {
        double scalar = 0.0;
            
        for (int i = 0; i < loss_wrt_weightedsum->columns; i++) {
            // Get the scalar once
            scalar = loss_wrt_weightedsum->data[input_index]->elements[i];
            
            for (int j = 0; j < wsum_wrt_weight->columns; j++) {
                double product_result = scalar * wsum_wrt_weight->data[input_index]->elements[j];
            
                // Accumulate the gradients by adding to the existing gradients
                double current_gradient = network->weight_gradients[layer_index]->data[i]->elements[j];
                double new_gradient = current_gradient + product_result;
                
                network->weight_gradients[layer_index]->data[i]->elements[j] = new_gradient;
            }
        }
    }
}

double accuracy(Matrix* targets, Matrix* outputs) {
    int counter = 0;
    // todo: change this back to output.rows
    for(int i = 0; i < outputs->rows; i++){
        int maxIndexOutputs = arg_max(outputs->data[i]);
        int maxIndexTargets = arg_max(targets->data[i]);
   
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

    for(int i = 0; i< network->num_layers; i++) {
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
            i, layer->num_neurons, layer->weights->columns, "he initializer", -.5, .5, 
            i < network->num_layers - 1 ? "leaky relu" : "softmax",
            (i == network->num_layers - 1) ? "" : ",\n"
        );
    }

    sprintf(json_output + strlen(json_output), // add to the end of the current string
        "\t\t\t]\n"
        "\t}\n"
    );

    OptimizationConfig* opt_config = network->optimization_config/* get your optimization_config object here */;

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
        opt_config->use_gradient_clipping,
        opt_config->use_gradient_clipping == 1 ? opt_config->gradient_clip_lower_bound : 0,
        opt_config->use_gradient_clipping == 1 ? opt_config->gradient_clip_upper_bound : 0,
        opt_config->use_learning_rate_decay,
        opt_config->learning_rate_decay_amount,
        opt_config->use_momentum,
        opt_config->momentum,
        get_optimizer_name(opt_config->optimizer),
        opt_config->epsilon,
        opt_config->rho,
        opt_config->adam_beta1,
        opt_config->adam_beta2
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
    for (int i = 0; i < network->num_layers; i++) {
        free_layer(network->layers[i]);

        free_matrix(network->weight_gradients[i]);
        free(network->weight_gradients[i]);
        
        free_vector(network->bias_gradients[i]);
        free(network->bias_gradients[i]);

        free_matrix(network->weighted_sums[i]);
        free(network->weighted_sums[i]);

        free_matrix(network->batched_outputs[i]);
        free(network->batched_outputs[i]);
    }

    free(network->weighted_sums);
    free(network->batched_outputs);
    free(network->layers);
    free(network->weight_gradients);
    free(network->bias_gradients);

    // free(network->optimization_config);
    
    // Finally, free the network itself
    free(network);
}

void free_network_config(NetworkConfig* config) { 
    free(config->neurons_per_layer);
}
char* serialize_optimization_config(OptimizationConfig* config) {
     if (config == NULL) {
        printf("Config is NULL\n");
        return NULL;
    }

    if(config->use_gradient_clipping == 0) {
        config->gradient_clip_lower_bound = 0.0f;
        config->gradient_clip_upper_bound = 0.0f;
    }

    if(config->optimizer != SGD){
        config->use_momentum = 0;
        config->momentum = 0.0f;
    }
    
    if(config->optimizer != RMS_PROP) {
        config->rho = 0.0f;
    }

    if(config->optimizer != ADAM) {
        config->adam_beta1 = 0.0f;
        config->adam_beta2 = 0.0f;
    }
    cJSON *root = cJSON_CreateObject();

    cJSON_AddItemToObject(root, "use_gradient_clipping", cJSON_CreateNumber(config->use_gradient_clipping));
    
    cJSON_AddItemToObject(root, "gradient_clip_lower_bound", cJSON_CreateNumber(config->gradient_clip_lower_bound));
    cJSON_AddItemToObject(root, "gradient_clip_upper_bound", cJSON_CreateNumber(config->gradient_clip_upper_bound));
    cJSON_AddItemToObject(root, "use_learning_rate_decay", cJSON_CreateNumber(config->use_learning_rate_decay));
    cJSON_AddItemToObject(root, "learning_rate_decay_amount", cJSON_CreateNumber(config->learning_rate_decay_amount));
    cJSON_AddItemToObject(root, "use_momentum", cJSON_CreateNumber(config->use_momentum));
    cJSON_AddItemToObject(root, "momentum", cJSON_CreateNumber(config->momentum));
    cJSON_AddItemToObject(root, "optimizer", cJSON_CreateNumber(config->optimizer));
    cJSON_AddItemToObject(root, "epsilon", cJSON_CreateNumber(config->epsilon));
    cJSON_AddItemToObject(root, "rho", cJSON_CreateNumber(config->rho));
    cJSON_AddItemToObject(root, "adam_beta1", cJSON_CreateNumber(config->adam_beta1));
    cJSON_AddItemToObject(root, "adam_beta2", cJSON_CreateNumber(config->adam_beta2));

    char *jsonString = cJSON_PrintUnformatted(root);

    cJSON_Delete(root);

    return jsonString;
}

char* serialize_network(const NNetwork* network) {
    cJSON *root = cJSON_CreateObject();
    cJSON *layers = cJSON_CreateArray();

    for (int i = 0; i < network->num_layers; i++) {
        char *layerString = serialize_layer(network->layers[i]);
        cJSON_AddItemToArray(layers, cJSON_CreateRaw(layerString));
        free(layerString);
    }

    cJSON_AddItemToObject(root, "num_layers", cJSON_CreateNumber(network->num_layers));
    cJSON_AddItemToObject(root, "layers", layers);
    cJSON_AddItemToObject(root, "loss_function", cJSON_CreateString(loss_fn_to_string(network->loss_fn)));
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

    network->num_layers = cJSON_GetObjectItem(json, "num_layers")->valueint;
    network->layers = malloc(network->num_layers * sizeof(Layer));
    network->weighted_sums = create_matrix_arr(network->num_layers);

    network->weight_gradients = create_matrix_arr(network->num_layers);
    
    network->bias_gradients = create_vector_arr(network->num_layers);

    network->batched_outputs = create_matrix_arr(network->num_layers);

    cJSON* json_layers = cJSON_GetObjectItem(json, "layers");
   
    for (int i = 0; i < network->num_layers; i++) {
        cJSON* json_layer = cJSON_GetArrayItem(json_layers, i);
        network->layers[i] = deserialize_layer(json_layer);
    }

    network->loss_fn = get_loss_fn_by_name(cJSON_GetObjectItem(json, "loss_function")->valuestring);
    network->optimization_config = NULL;

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