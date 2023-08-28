#include "nnetwork.h"
#include "activation_function.h"
#include "loss_functions.h"
#include <string.h>
#include <time.h>

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
    network->num_layers = config->numLayers;
    network->layers = malloc(network->num_layers * sizeof(Layer));

    for (int i = 0; i < config->numLayers; i++) {
        LayerConfig layerConfig;
        layerConfig.num_inputs = i == 0 ? config->num_features : network->layers[i - 1]->num_neurons;
        layerConfig.num_neurons = config->neurons_per_layer[i];
        layerConfig.activation_fn = config->activation_fns[i];

        if(i < config->numLayers - 1){
            int use_regularization = 0;

            if(config->weight_lambdas != NULL && config->weight_lambdas->size > 0){
                use_regularization = 1;
                layerConfig.weight_lambda = config->weight_lambdas->elements[i]; 
            }else {
                layerConfig.weight_lambda = 0;
            }

            if(config->bias_lambdas != NULL && config->bias_lambdas->size > 0) {
                use_regularization = 1;
                layerConfig.bias_lambda = config->bias_lambdas->elements[i];
            }else {
                layerConfig.bias_lambda = 0;
            }

            layerConfig.use_regularization = use_regularization;
        }
        Layer* layer = create_layer(&layerConfig);
        network->layers[i] = layer;
    }
    

    network->loss_fn = config->loss_fn;

    network->optimization_config = config->optimization_config;

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
    init_network_memory(network, config->num_rows);
    
    log_info("%s", "Created Network:");
    dump_network_config(network);

    return network;
}

void init_network_memory(NNetwork *network, int num_rows) {
    network->layer_outputs = create_matrix_arr(network->num_layers);
    network->weighted_sums = create_matrix_arr(network->num_layers);
    network->weight_gradients = create_matrix_arr(network->num_layers);
    network->bias_gradients = create_vector_arr(network->num_layers);

    for(int i = 0; i < network->num_layers; i++) {
        // I allocate memory for vectors using calloc so they are already initialized to zeroes!
        network->layer_outputs[i] = create_matrix(num_rows, network->layers[i]->num_neurons);
        network->weighted_sums[i] = create_matrix(num_rows, network->layers[i]->num_neurons);
        network->weight_gradients[i] = create_matrix(network->layers[i]->weights->rows, network->layers[i]->weights->columns);
        network->bias_gradients[i] = create_vector(network->layers[i]->biases->size);
    }
}
/*
    This method performs the forward pass of a neural network using a batched processing approach. 
    It processes the entire input batch in a vectorized manner, taking advantage of parallelism and optimized matrix operations.

    Use this only if your computer can support multithreading and/or CUDA.
*/
void forward_pass_batched(NNetwork* network, Matrix* input_matrix) { 
    for(int layer_index = 0; layer_index < network->num_layers; layer_index++) {

        Layer* current_layer = network->layers[layer_index];

        Matrix* transposed_weights = matrix_transpose(current_layer->weights);
        
        // do not free this, it points to the original input matrix for the first layer!
        Matrix* layer_input = layer_index == 0 ? input_matrix : network->layer_outputs[layer_index - 1];

        #ifdef CUDA_ENABLED
            Matrix* product_result = matrix_product_cuda(layer_input, transposed_weights);

            matrix_vector_addition_cuda(product_result, current_layer->biases, network->weighted_sums[layer_index]);

            copy_matrix_cuda(network->weighted_sums[layer_index], network->layer_outputs[layer_index]);
        #else
            Matrix* product_result = matrix_product(layer_input, transposed_weights);

            matrix_vector_addition(product_result, current_layer->biases, network->weighted_sums[layer_index]);
            
            copy_matrix_into(network->weighted_sums[layer_index], network->layer_outputs[layer_index]);
        #endif

        // log_info("output: %s", matrix_to_string(network->layer_outputs[layer_index]));

        switch(current_layer->activation_fn) {
            case LEAKY_RELU:
                leakyRelu(network->layer_outputs[layer_index]->data);
                break;
            case SOFTMAX:
                softmax_matrix(network->layer_outputs[layer_index]);
                break;
            default:
                log_error("Unknown Activation Function: %s! \n be sure to register it to the workflow.", get_activation_function_name(current_layer->activation_fn));
                break;
        }

        free_matrix(product_result);
        free_matrix(transposed_weights);
    }
}

void calculate_loss(NNetwork* network, Matrix* yValues) {
    switch(network->loss_fn) {
        case MEAN_SQUARED_ERROR:
            // network->loss = meanSquaredError(Matrix *outputs, Matrix *targets)
            break;
        case CATEGORICAL_CROSS_ENTROPY:
            network->loss = calculateCategoricalCrossEntropyLoss(yValues, network->layer_outputs[network->num_layers - 1]);
            break;
        case UNRECOGNIZED_LFN:
            log_error("Unrecognized Loss Function!");
            return;
    }

    network->accuracy = accuracy(yValues, network->layer_outputs[network->num_layers - 1]);
}

void backpropagation_batched(NNetwork* network, Matrix* input_matrix, Matrix* y_values) {
    #ifdef DEBUG
        char* debug_str;
    #endif
    int num_layers = network->num_layers;
    Matrix** loss_wrt_weightedsum = create_matrix_arr(num_layers);

    // -------------OUTPUT LAYER-------------
    int layer_index = num_layers - 1;
    Matrix* output = network->layer_outputs[layer_index];

    // I can distribute the work amongst the threads in the thread pool for all three operations.
    Matrix* loss_wrt_output = create_matrix(y_values->rows, y_values->columns);
    computeCategoricalCrossEntropyLossDerivativeMatrix(y_values, output, loss_wrt_output);

    // printf("%s \n", matrix_to_string(loss_wrt_output));
    #ifdef DEBUG
        debug_str = matrix_to_string(loss_wrt_output);
        log_info("loss_wrt_output: %s", debug_str);
        free(debug_str);
    #endif

    // TODO: ABSTRACT THIS OUT TO WORK WITH ANY ACTIVATION FUNCTION
    // jacobian_matrices AKA output_wrt_weightedsum
    Matrix** jacobian_matrices = softmax_derivative_batched(output, network->thread_pool);

    loss_wrt_weightedsum[layer_index] = batch_matrix_vector_product(jacobian_matrices, loss_wrt_output, output->rows);
    #ifdef DEBUG
        debug_str = matrix_to_string(loss_wrt_weightedsum[layer_index]);
        log_info("Loss wrt WeightedSum matrix for layer #%d: %s", layer_index, debug_str);
        free(debug_str);
    #endif

    Matrix* weightedsum_wrt_weight = NULL;    

    if(layer_index == 0) {
        weightedsum_wrt_weight = input_matrix;
    }else {
        weightedsum_wrt_weight = network->layer_outputs[layer_index - 1];
    }
    
    #ifdef DEBUG
        debug_str = matrix_to_string(weightedsum_wrt_weight);
        log_info("weightedsum wrt weights for layer #%d: %s", layer_index, debug_str);
        free(debug_str);
    #endif

    // clock_t start_wgradients = clock();
    // multiplying each weighted sum with different input neurons to get the gradients of the weights that connect them
    for (int input_index = 0; input_index < weightedsum_wrt_weight->rows; input_index++) {
        double scalar = 0.0; // Initialize scalar to zero
        
        for (int i = 0; i < loss_wrt_weightedsum[layer_index]->columns; i++) {
            // Get the scalar once
            scalar= loss_wrt_weightedsum[layer_index]->get_element(loss_wrt_weightedsum[layer_index], input_index, i);
            
            for (int j = 0; j < weightedsum_wrt_weight->columns; j++) {
                double product_result = scalar * weightedsum_wrt_weight->get_element(weightedsum_wrt_weight, input_index, j);
                
                // Update the gradients directly without creating additional vectors
                double new_gradient = network->weight_gradients[layer_index]->get_element(network->weight_gradients[layer_index], i, j) + product_result;
                network->weight_gradients[layer_index]->set_element(network->weight_gradients[layer_index], i, j, new_gradient);
            }
        }
    }
    // clock_t end_wgradients = clock();
    // double wgradients_time_elapsed = ((double)(end_wgradients - start_wgradients) / CLOCKS_PER_SEC) * 1000;
    // log_info("weight gradient calculations elapsed time: %f", wgradients_time_elapsed);

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

    // clean memory
    free_matrix(loss_wrt_output);
    
    for(int i = 0; i < output->rows; i++) {
        free_matrix(jacobian_matrices[i]);
    }
    free(jacobian_matrices);
    
    // ------------- HIDDEN LAYERS -------------
    // we do need to iterate over other layers
    for (layer_index -= 1; layer_index >= 0; layer_index--) { // current layer's dimensions = (4 inputs, 4 neurons)
        #ifdef CUDA_ENABLED
            Matrix* loss_wrt_output = matrix_product_cuda(loss_wrt_weightedsum[layer_index + 1], network->layers[layer_index + 1]->weights);
        #else
            Matrix* loss_wrt_output = matrix_product(loss_wrt_weightedsum[layer_index + 1], network->layers[layer_index + 1]->weights);
        #endif
        
        #ifdef DEBUG
            char* log_wrt_output_str = matrix_to_string(loss_wrt_output);
            log_info("loss wrt output for layer: #%d: %s", layer_index, log_wrt_output_str);
            free(log_wrt_output_str);
        #endif
        Matrix* output_wrt_weightedsums = NULL;

        switch(network->layers[layer_index]->activation_fn) {
            case RELU:
                log_info("not implemented yet!");
                break;
            case LEAKY_RELU:
                output_wrt_weightedsums = leakyRelu_derivative_matrix(network->weighted_sums[layer_index]);
                break;
            case SOFTMAX: // 
                log_error("cannot/shouldn't be softmax in the hidden layers.");
                break;
            case UNRECOGNIZED_AFN:
                log_error("Unrecognized activation function!");
                return;
        }

        if(output_wrt_weightedsums == NULL) {
            log_error("Failed to calculate output_wrt_weightedsums, please check previous logs!");
        }

        #ifdef DEBUG
            char* debug_str = matrix_to_string(output_wrt_weightedsums);
            log_info("output wrt wsum for layer #%d: %s", layer_index, debug_str);
            free(debug_str);
        #endif
        
        #ifdef CUDA_ENABLED
            loss_wrt_weightedsum[layer_index] = matrix_element_wise_operation_cuda(loss_wrt_output, output_wrt_weightedsums, MULTIPLY);
        #else
            loss_wrt_weightedsum[layer_index] = matrix_multiplication(loss_wrt_output, output_wrt_weightedsums);
        #endif        
        free_matrix(loss_wrt_output);

        free_matrix(output_wrt_weightedsums);
        
        #ifdef DEBUG
            debug_str = matrix_to_string(loss_wrt_weightedsum[layer_index]);
            log_info("loss wrt weighted sum for layer #%d: %s", layer_index, debug_str);
            free(debug_str);
        #endif

        if (layer_index == 0) {
            weightedsum_wrt_weight = input_matrix;
        } else {
            weightedsum_wrt_weight = network->layer_outputs[layer_index - 1];
        }

        // log_info("weightedsum wrt weights for layer #%d: %s", layer_index, matrix_to_string(weightedsum_wrt_weight));
        
        // clock_t hidden_layer_wg_start = clock();

        // multiplying each weighted sum with different input neurons to get the gradients of the weights that connect them
        for (int input_index = 0; input_index < weightedsum_wrt_weight->rows; input_index++) {
            double scalar = 0.0; // Initialize scalar to zero
            
            for (int i = 0; i < loss_wrt_weightedsum[layer_index]->columns; i++) {
                // Get the scalar once
                scalar= loss_wrt_weightedsum[layer_index]->get_element(loss_wrt_weightedsum[layer_index], input_index, i);
                
                for (int j = 0; j < weightedsum_wrt_weight->columns; j++) {
                    double product_result = scalar * weightedsum_wrt_weight->get_element(weightedsum_wrt_weight, input_index, j);
                    
                    // Update the gradients directly without creating additional vectors
                    double new_gradient = network->weight_gradients[layer_index]->get_element(network->weight_gradients[layer_index], i, j) + product_result;
                    network->weight_gradients[layer_index]->set_element(network->weight_gradients[layer_index], i, j, new_gradient);
                }
            }
        }
        #ifdef DEBUG
            debug_str = matrix_to_string(network->weight_gradients[layer_index]);
            log_info("Weight gradients of the layer #%d: %s", layer_index, debug_str);
            free(debug_str);
        #endif
        // clock_t hidden_layer_wg_end = clock();
        // log_info("time it took by weight gradient calculation of hidden layer: %f", ((double)(hidden_layer_wg_end - hidden_layer_wg_start) / CLOCKS_PER_SEC) * 1000);

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

    for(int i = 0; i < network->num_layers; i++) {
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

    OptimizationConfig* opt_config = network->optimization_config/* get your OptimizationConfig object here */;

    sprintf(
        json_output + strlen(json_output),
        "\t\t\"Optimizer Config\": {\n"
        "\t\t\t\"use_gradient_clipping\": %d,\n"
        "\t\t\t\"gradient_clip_lower_bound\": %f,\n"
        "\t\t\t\"gradient_clip_upper_bound\": %f,\n"
        "\t\t\t\"use_learning_rate_decay\": %d,\n"
        "\t\t\t\"learning_rate_decay_amount\": %f,\n"
        "\t\t\t\"use_momentum\": %d,\n"
        "\t\t\t\"momentum\": %f,\n"
        "\t\t\t\"optimizer\": %s,\n"
        "\t\t\t\"epsilon\": %f,\n"
        "\t\t\t\"rho\": %f,\n"
        "\t\t\t\"adam_beta1\": %f,\n"
        "\t\t\t\"adam_beta2\": %f\n"
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
        
        free_vector(network->bias_gradients[i]);

        free_matrix(network->weighted_sums[i]);

        free_matrix(network->layer_outputs[i]);
    }

    free(network->weighted_sums);
    free(network->layer_outputs);
    free(network->layers);
    free(network->weight_gradients);
    free(network->bias_gradients);
    
    // Free the loss function
    free(network->optimization_config);

    // Finally, free the network itself
    free(network);
}

void free_network_config(NetworkConfig* config) { 
    free(config->neurons_per_layer);

    free_vector(config->weight_lambdas);
    free_vector(config->bias_lambdas);
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
    cJSON_AddItemToObject(root, "num_threads", cJSON_CreateNumber(network->thread_pool->num_threads));

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
    
    cJSON* json_layers = cJSON_GetObjectItem(json, "layers");
   
    for (int i = 0; i < network->num_layers; i++) {
        cJSON* json_layer = cJSON_GetArrayItem(json_layers, i);
        network->layers[i] = deserialize_layer(json_layer);
    }

    network->optimization_config = NULL;

    network->loss_fn = get_loss_fn_by_name(cJSON_GetObjectItem(json, "loss_function")->valuestring);
    // network->loss = cJSON_GetObjectItem(json, "loss")->valuedouble;
    // network->accuracy = cJSON_GetObjectItem(json, "accuracy")->valuedouble;
    int thread_count = cJSON_GetObjectItem(json, "num_threads")->valueint;
    network->thread_pool = create_thread_pool(thread_count);
        
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