#include "nnetwork.h"

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
        layerConfig.inputSize = i == 0 ? config->inputSize : network->layers[i - 1]->neuronCount;
        layerConfig.neuronCount = config->neuronsPerLayer[i];
        layerConfig.activationFunction = &config->activationFunctions[i];

        if(i < config->numLayers - 1){
            int use_regularization = 0;

            if(config->weightLambdas != NULL && config->weightLambdas->size > 0){
                use_regularization = 1;
                layerConfig.weightLambda = config->weightLambdas->elements[i]; 
            }

            if(config->biasLambdas != NULL && config->biasLambdas->size > 0) {
                use_regularization = 1;
                layerConfig.biasLambda = config->biasLambdas->elements[i];
            }

            log_debug("should use regularization? %d", use_regularization);
            layerConfig.shouldUseRegularization = use_regularization;
        }
        Layer* layer = create_layer(&layerConfig);
        network->layers[i] = layer;
    }
    

    network->lossFunction = config->lossFunction;
    network->optimizationConfig = config->optimizationConfig;

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


    log_info("Created Network:");
    dump_network_config(network);

    return network;
}

void forward_pass(NNetwork* network, Vector* input, Vector* output) {
    network->layers[0]->input = input;
    for (int layerIndex = 0; layerIndex < network->layerCount; layerIndex++) {
        Layer* currentLayer = network->layers[layerIndex];
       
        Vector* dotProduct = dot_product(currentLayer->weights, currentLayer->input);
        #ifdef DEBUG
            char* weightsStr = matrix_to_string(currentLayer->weights);
            char* dotProductStr = vector_to_string(dotProduct);
            log_debug(
                "Weights For Input@Layer: %d: %s", 
                layerIndex, 
                weightsStr
            );

            log_debug(
                "Dot Product For Input@Layer: %d: %s", 
                layerIndex, 
                dotProductStr
            );

            free(weightsStr);
            free(dotProductStr);
        #endif
        
        currentLayer->output = vector_addition(dotProduct, currentLayer->biases);

        #ifdef DEBUG
            char* netInput = vector_to_string(currentLayer->output);
            log_debug(
                "Net Input For Input Row's Layer: %d: %s",  
                layerIndex, 
                netInput
            );
            free(netInput);
        #endif
        currentLayer->weightedSums = copy_vector(currentLayer->output);
            
        currentLayer->activationFunction->activation(currentLayer->output);
            
        #ifdef DEBUG
            char* outputStr = vector_to_string(currentLayer->output);
            log_debug(
                "Output of activation function for Input Row for Layer: %d: %s", 
                layerIndex, 
                outputStr
            );
            free(outputStr);
        #endif

        if(layerIndex != network->layerCount - 1) {
            network->layers[layerIndex + 1]->input = currentLayer->output;
        }

    }
    // copy out the outputs
    for(int i = 0; i < output->size; i++) {
        output->elements[i] = network->layers[network->layerCount - 1]->output->elements[i];
    }
    #ifdef DEBUG
        log_info("Completed forward pass.");
    #endif
}

void calculate_loss(NNetwork* network, Matrix* yValues) {
    network->loss = categoricalCrossEntropyLoss(yValues, network->output);

    for(int i = 0; i < network->layerCount; i++) {
        Layer* layer = network->layers[i];
        double lambda = layer->weightLambda;
        double weightPenalty = 0.0f;

        for(int row = 0; row < layer->weights->rows; row ++) {
            weightPenalty += calculate_l1_penalty(lambda, layer->weights->data[row]);
            weightPenalty += calculate_l2_penalty(lambda, layer->weights->data[row]);
        }

        double biasPenalty = calculate_l1_penalty(lambda, layer->biases) + calculate_l2_penalty(lambda, layer->biases);

        network->loss += weightPenalty + biasPenalty;
    }

    network->accuracy = accuracy(yValues, network->output);
}

void backpropagation(NNetwork* network, Vector* input, Vector* output, Vector* target) {
    #ifdef DEBUG
        log_info("Start of backward pass.");
    #endif
    // for each output
    // todo: change this back to output.rows
        Vector* prediction = output;
        
        // @todo: try to abstract this out.
        Vector* dLoss_dOutputs = categoricalCrossEntropyLossDerivative(target, prediction);

        Matrix* jacobian = softmax_derivative(prediction);
        Vector* dLoss_dWeightedSums = dot_product(jacobian, dLoss_dOutputs);

        /* These calculations are verified to be correct, but you can comment out the log statements
           to check again.
        
            #ifdef DEBUG
                log_debug("jacobian matrix: %s", matrix_to_string(jacobian));
                log_debug("dLoss/dWeightedSums is: %s", vector_to_string(dLoss_dWeightedSums));
            #endif
        */
       
        // Calculating dLoss/dWeights and dLoss/dInputs for the output layer
        int layerIndex = network->layerCount - 1;
        Layer* currentLayer = network->layers[layerIndex];

        for(int outputNeuronIndex = 0; outputNeuronIndex < currentLayer->neuronCount; outputNeuronIndex++) {
            #ifdef DEBUG
                log_debug("Partial derivative of Loss with respect to Output for target: %f, prediction: %f is %f \n", target->elements[outputNeuronIndex], prediction->elements[outputNeuronIndex], dLoss_dOutputs->elements[outputNeuronIndex]);
            #endif

            #ifdef DEBUG
                log_debug("Partial derivative of Loss with respect to the input to the activation function of the output layer's node #%d is: %f \n",outputNeuronIndex ,dLoss_dWeightedSums->elements[outputNeuronIndex]);
            #endif

            for(int weightIndex = 0; weightIndex < currentLayer->weights->columns; weightIndex++) {
                double dWeightedSum_dWeight = 0.0f;
                if(layerIndex == 0) {
                    dWeightedSum_dWeight= input->elements[weightIndex];
                }else {
                    dWeightedSum_dWeight = network->layers[layerIndex-1]->output->elements[weightIndex];
                }

                double dLoss_dWeight = dLoss_dWeightedSums->elements[outputNeuronIndex] * dWeightedSum_dWeight;

                #ifdef DEBUG
                    log_debug(
                        "Partial derivative calculations for weight #%d of the neuron #%d: \n"
                        "\t\t\t\tdLoss_dWeightedSum: %f \n"
                        "\t\t\t\tdWeightedSum_dWeight: %f \n"
                        "\t\t\t\tdLoss_dWeight: %f \n", 
                        weightIndex, outputNeuronIndex, dLoss_dWeightedSums->elements[outputNeuronIndex], dWeightedSum_dWeight, dLoss_dWeight
                    );
                #endif

                currentLayer->gradients->data[outputNeuronIndex]->elements[weightIndex] += dLoss_dWeight;
                
                // Gradient Clipping
                if(network->optimizationConfig->shouldUseGradientClipping == 1) {
                    double originalGradient = currentLayer->gradients->data[outputNeuronIndex]->elements[weightIndex];
                    if(originalGradient < network->optimizationConfig->gradientClippingLowerBound) {
                        currentLayer->gradients->data[outputNeuronIndex]->elements[weightIndex] = network->optimizationConfig->gradientClippingLowerBound;
                    } else if(originalGradient > network->optimizationConfig->gradientClippingUpperBound) {
                        currentLayer->gradients->data[outputNeuronIndex]->elements[weightIndex] = network->optimizationConfig->gradientClippingUpperBound;
                    }
                    double clippedGradient = currentLayer->gradients->data[outputNeuronIndex]->elements[weightIndex];
                    
                    #ifdef DEBUG
                        if(originalGradient != clippedGradient) {
                            log_debug(
                                "Gradient clipping applied for weight #%d of the neuron #%d: \n"
                                "\t\t\t\tOriginal gradient: %f \n"
                                "\t\t\t\tClipped gradient: %f \n", 
                                weightIndex, outputNeuronIndex, originalGradient, clippedGradient
                            );
                        }
                    #endif
                }
            }

            currentLayer->biasGradients->elements[outputNeuronIndex] = dLoss_dWeightedSums->elements[outputNeuronIndex];

            // Gradient Clipping
            if(network->optimizationConfig->shouldUseGradientClipping == 1) {
                double originalBiasGradient = currentLayer->biasGradients->elements[outputNeuronIndex] = dLoss_dWeightedSums->elements[outputNeuronIndex];
                if(originalBiasGradient < network->optimizationConfig->gradientClippingLowerBound) {
                    currentLayer->biasGradients->elements[outputNeuronIndex] = network->optimizationConfig->gradientClippingLowerBound;
                } else if(originalBiasGradient > network->optimizationConfig->gradientClippingUpperBound) {
                    currentLayer->biasGradients->elements[outputNeuronIndex] = network->optimizationConfig->gradientClippingUpperBound;
                }
                double clippedBiasGradient = currentLayer->biasGradients->elements[outputNeuronIndex];
                
                #ifdef DEBUG
                    if(originalBiasGradient != clippedBiasGradient) {
                        log_debug(
                            "Bias gradient clipping applied for neuron #%d: \n"
                            "\t\t\t\tOriginal bias gradient: %f \n"
                            "\t\t\t\tClipped bias gradient: %f \n", 
                            outputNeuronIndex, originalBiasGradient, clippedBiasGradient
                        );
                    }
                #endif
            }

            // Backpropagating the error to the hidden layers
            currentLayer->dLoss_dWeightedSums->elements[outputNeuronIndex] = dLoss_dWeightedSums->elements[outputNeuronIndex];
            #ifdef DEBUG
                log_debug(
                    "Partial derivative of Loss with respect to the weighted sum for neuron #%d: %f \n", 
                    outputNeuronIndex, dLoss_dWeightedSums->elements[outputNeuronIndex]
                );
            #endif
        }
        // log_info("Gradient for weights and biases computed for the output layer.");

        #ifdef DEBUG
            log_debug(
                "Gradients for layer #%d: \n"
                "\t\t\t\tWeight gradients: \n%s\n"
                "\t\t\t\tBias gradients: %s\n",
                layerIndex, matrix_to_string(currentLayer->gradients), vector_to_string(currentLayer->biasGradients)
            );
        #endif
        
        free_matrix(jacobian);
        free_vector(dLoss_dOutputs);
        free_vector(dLoss_dWeightedSums);

        #ifdef DEBUG
            log_debug("Calculating gradients for the hidden layers of output index: %d", outputIndex);
        #endif
        for(layerIndex = network->layerCount - 2; layerIndex >= 0; layerIndex --) {
            // log_info("Backward propagation for layer index: %d", layerIndex);
            currentLayer = network->layers[layerIndex];
            Vector* l1_bias_derivatives;
            Vector* l2_bias_derivatives;

            if(currentLayer->biasLambda > 0 ){
                l1_bias_derivatives = l1_derivative(currentLayer->biasLambda, currentLayer->biases);
                l2_bias_derivatives = l2_derivative(currentLayer->biasLambda, currentLayer->biases);
            }

            for(int neuronIndex = 0; neuronIndex < currentLayer->neuronCount; neuronIndex++) {
                double dLoss_dOutput = 0.0f;
                
                // this calculation is correct
                Layer* nextLayer = network->layers[layerIndex + 1];
                for(int neuronIndexNext = 0; neuronIndexNext < nextLayer->neuronCount; neuronIndexNext++) {
                    dLoss_dOutput += nextLayer->dLoss_dWeightedSums->elements[neuronIndexNext] * nextLayer->weights->data[neuronIndexNext]->elements[neuronIndex];
                }

                #ifdef DEBUG
                    log_debug("Partial derivative of Loss with respect to Output is: %f", dLoss_dOutput);
                #endif
                
                double dOutput_dWeightedSum = currentLayer->activationFunction->derivative(currentLayer->weightedSums->elements[neuronIndex]);
                double dLoss_dWeightedSum = dLoss_dOutput * dOutput_dWeightedSum;

                #ifdef DEBUG
                    log_debug("Partial derivative of Output with respect to Net Input is: %f", dOutput_dWeightedSum);
                    log_debug("Partial derivative of Loss with respect to Net Input is: %f", dLoss_dWeightedSum);
                #endif
                Vector* l1_weights_derivatives;
                Vector* l2_weights_derivatives;

                if(currentLayer->weightLambda > 0) {
                    l1_weights_derivatives = l1_derivative(currentLayer->weightLambda, currentLayer->weights->data[neuronIndex]);
                    l2_weights_derivatives = l2_derivative(currentLayer->weightLambda, currentLayer->weights->data[neuronIndex]);
                }

                for(int weightIndex = 0; weightIndex < currentLayer->weights->columns; weightIndex++) {
                    double dWeightedSum_dWeight = 0.0f;
                    if(layerIndex == 0) {
                        dWeightedSum_dWeight= input->elements[weightIndex];
                    }else {
                        dWeightedSum_dWeight = network->layers[layerIndex-1]->output->elements[weightIndex];
                    }
                    
                    
                    double dLoss_dWeight = dLoss_dWeightedSum * dWeightedSum_dWeight;
                    if(currentLayer->weightLambda > 0) {
                        dLoss_dWeight += l1_weights_derivatives->elements[weightIndex];
                        dLoss_dWeight += l2_weights_derivatives->elements[weightIndex];
                    }

                    #ifdef DEBUG
                        log_debug(
                            "Partial derivative calculations for weight #%d of the neuron #%d: \n"
                            "\t\t\t\tdLoss_dWeightedSum: %f \n"
                            "\t\t\t\tdWeightedSum_dWeight: %f \n"
                            "\t\t\t\tdLoss_dWeight: %f \n", 
                            weightIndex, neuronIndex, dLoss_dWeightedSum, dWeightedSum_dWeight, dLoss_dWeight
                        );
                    #endif

                    currentLayer->gradients->data[neuronIndex]->elements[weightIndex] += dLoss_dWeight;
                }
                
                currentLayer->biasGradients->elements[neuronIndex] = dLoss_dWeightedSum;

                
                if(currentLayer->biasLambda > 0) {
                    currentLayer->biasGradients->elements[neuronIndex] += l1_bias_derivatives->elements[neuronIndex] + l2_bias_derivatives->elements[neuronIndex];
                    
                }

                

                if(network->optimizationConfig->shouldUseGradientClipping == 1) {
                    double originalBiasGradient = currentLayer->biasGradients->elements[neuronIndex];
            
                    if (originalBiasGradient < network->optimizationConfig->gradientClippingLowerBound) {
                        currentLayer->biasGradients->elements[neuronIndex] = network->optimizationConfig->gradientClippingLowerBound;
                    } else if (originalBiasGradient > network->optimizationConfig->gradientClippingUpperBound) {
                        currentLayer->biasGradients->elements[neuronIndex] = network->optimizationConfig->gradientClippingUpperBound;
                    }
                    
                    double clippedBiasGradient = currentLayer->biasGradients->elements[neuronIndex];

                    #ifdef DEBUG
                        if (originalBiasGradient != clippedBiasGradient) {
                            log_debug(
                                "Bias gradient clipping applied for neuron #%d: \n"
                                "\t\t\t\tOriginal bias gradient: %f \n"
                                "\t\t\t\tClipped bias gradient: %f \n", 
                                neuronIndex, originalBiasGradient, clippedBiasGradient
                            );
                        }
                    #endif
                }

                currentLayer->dLoss_dWeightedSums->elements[neuronIndex] = dLoss_dOutput * dOutput_dWeightedSum;

                // log_info("Gradient for weights and biases computed for neuron index: %d in layer index: %d", neuronIndex, layerIndex);
                if(currentLayer->weightLambda > 0) {
                    free_vector(l1_weights_derivatives);
                    free_vector(l2_weights_derivatives);
                }
            }
            if(currentLayer->biasLambda > 0) {
                free_vector(l1_bias_derivatives);
                free_vector(l2_bias_derivatives);
            }
           
            #ifdef DEBUG
                log_debug(
                    "Gradients for layer #%d: \n"
                    "\t\t\t\tWeight gradients: \n%s\n"
                    "\t\t\t\tBias gradients: %s\n",
                    layerIndex, matrix_to_string(currentLayer->gradients), vector_to_string(currentLayer->biasGradients)
                );
            #endif
            // log_info("Gradient computation complete for current hidden layer at index: %d", layerIndex);
        }
        // log_info("Backward pass complete for current output row.");
    
    #ifdef DEBUG
        log_info("End of backward pass.");
    #endif
}

// void delete_network(NNetwork* network){
//     for(int i = network->layerCount - 1; i >= 0; i--) {
//         free_layer(network->layers[i]);
//     }
// }

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
        opt_config->gradientClippingLowerBound,
        opt_config->gradientClippingUpperBound,
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

double calculate_l1_penalty(double lambda, Vector* vector) {
    double penalty = 0.0f;

    for(int i = 0; i < vector->size; i++) {
        penalty += fabs(vector->elements[i]);
    }

    return lambda * penalty;
}

double calculate_l2_penalty(double lambda, Vector* vector) {
    double penalty = 0.0f;

    for(int i = 0; i < vector->size; i++) {
        penalty += pow(vector->elements[i], 2);
    }

    return lambda * penalty;
}

Vector* l1_derivative(double lambda, Vector* vector) {
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
Vector* l2_derivative(double lambda, Vector* vector) {
    Vector* derivatives = create_vector(vector->size);

    for(int i = 0; i < derivatives->size; i++) {
        derivatives->elements[i] = 2 * lambda * vector->elements[i];
    }

    return derivatives;
}

void delete_network(NNetwork* network) {
    if (network == NULL) {
        return;
    }

    // Free the layers
    for (int i = 0; i < network->layerCount; i++) {
        free_layer(network->layers[i]);
    }
    free(network->layers);

    // Free the loss function
    free(network->lossFunction);

    // Free the optimization configuration
    // free_OptimizationConfig(network->optimizationConfig);

    // Free the output matrix
    free_matrix(network->output);

    // Finally, free the network itself
    free(network);
}