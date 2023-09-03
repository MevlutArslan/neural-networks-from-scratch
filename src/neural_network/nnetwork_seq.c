#include "nnetwork.h"


void forward_pass_row_by_row(NNetwork* network, Vector* input, Vector* output) {
    network->layers[0]->input = input;
    for (int layerIndex = 0; layerIndex < network->num_layers; layerIndex++) {
        Layer* current_layer = network->layers[layerIndex];
        Vector* dotProduct = create_vector(current_layer->weights->rows);

        dot_product(current_layer->weights, current_layer->input, dotProduct);
        #ifdef DEBUG
            char* weightsStr = matrix_to_string(current_layer->weights);
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
        
        vector_addition(dotProduct, current_layer->biases, current_layer->output);
        
        free_vector(dotProduct);
        #ifdef DEBUG
            char* netInput = vector_to_string(currentLayer->output);
            log_debug(
                "Net Input For Input Row's Layer: %d: %s",  
                layerIndex, 
                netInput
            );
            free(netInput);
        #endif

        // I initialized it during create_layer() so I have to clear it before assigning a vector 
        // to it to prevent leaks 
        free_vector(current_layer->weighted_sums);
        current_layer->weighted_sums = copy_vector(current_layer->output);

        switch(current_layer->activation_fn) {
            case LEAKY_RELU:
                leakyRelu(current_layer->output);
                break;
            case SOFTMAX:
                softmax(current_layer->output);
                break;
            default:
                log_error("Unknown Activation Function: %s! \n be sure to register it to the workflow.", get_activation_function_name(current_layer->activation_fn));
                break;
        }

            
        #ifdef DEBUG
            char* outputStr = vector_to_string(currentLayer->output);
            log_debug(
                "Output of activation function for Input Row for Layer: %d: %s", 
                layerIndex, 
                outputStr
            );
            free(outputStr);
        #endif

        if(layerIndex != network->num_layers - 1) {
            network->layers[layerIndex + 1]->input = current_layer->output;
        }
    }
    // copy out the outputs
    for(int i = 0; i < output->size; i++) {
        output->elements[i] = network->layers[network->num_layers - 1]->output->elements[i];
    }
    #ifdef DEBUG
        log_info("Completed forward pass.");
    #endif
}


/**
 * Performs the backpropagation process in the neural network.
 * 
 * IMPORTANT: If a different activation function is used, the logic for calculating the gradients of 
 * the output layer needs to be modified accordingly. Specifically, the derivative function of the 
 * new activation function should be properly implemented and used in place of the existing derivative 
 * functions (e.g., softmax_derivative).
 *
 * @param network The neural network to perform backpropagation on.
 * @param input The input vector to the forward pass.
 * @param output The output of the forward pass.
 * @param target The vector with the expected outputs.
 */
void backpropagation(NNetwork* network, Vector* input, Vector* output, Vector* target) {
    #ifdef DEBUG
        log_info("Start of backward pass.");
    #endif
        Vector* prediction = output;
        
        Vector* dLoss_dOutputs = categoricalCrossEntropyLossDerivative(target, prediction);
        // log_info("dLoss_dOutputs: %s", vector_to_string(dLoss_dOutputs));

        Matrix* jacobian = softmax_derivative(prediction);
        // log_info("jacobian: %s", matrix_to_string(jacobian));

        Vector* dLoss_dWeightedSums = create_vector(jacobian->rows);
        dot_product(jacobian, dLoss_dOutputs, dLoss_dWeightedSums);
        // log_info("dloss_wrt_weightedsum: %s", vector_to_string(dLoss_dWeightedSums));

        /* These calculations are verified to be correct, but you can comment out the log statements
           to check again.
        
            #ifdef DEBUG
                char* jacobian_matrix_str = matrix_to_string(jacobian);
                char* dLoss_dWeightedSums_str = vector_to_string(dLoss_dWeightedSums);
                
                log_debug("jacobian matrix: %s", jacobian_matrix_str);
                log_debug("dLoss/dWeightedSums is: %s", dLoss_dWeightedSums_str);
                
                free(jacobian_matrix_str);
                free(dLoss_dWeightedSums_str);
            #endif
        */

        int layerIndex = network->num_layers - 1;
        Layer* currentLayer = network->layers[layerIndex];
        for(int outputNeuronIndex = 0; outputNeuronIndex < currentLayer->num_neurons; outputNeuronIndex++) {
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

                currentLayer->weight_gradients->data[outputNeuronIndex]->elements[weightIndex] += dLoss_dWeight;
                
                // Gradient Clipping
                if(network->optimization_config->use_gradient_clipping == 1) {
                    double originalGradient = currentLayer->weight_gradients->data[outputNeuronIndex]->elements[weightIndex];
                    if(originalGradient < network->optimization_config->gradient_clip_lower_bound) {
                        currentLayer->weight_gradients->data[outputNeuronIndex]->elements[weightIndex] = network->optimization_config->gradient_clip_lower_bound;
                    } else if(originalGradient > network->optimization_config->gradient_clip_upper_bound) {
                        currentLayer->weight_gradients->data[outputNeuronIndex]->elements[weightIndex] = network->optimization_config->gradient_clip_upper_bound;
                    }
                    double clippedGradient = currentLayer->weight_gradients->data[outputNeuronIndex]->elements[weightIndex];
                    
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

            currentLayer->bias_gradients->elements[outputNeuronIndex] += dLoss_dWeightedSums->elements[outputNeuronIndex];
  
            // Gradient Clipping
            if(network->optimization_config->use_gradient_clipping == 1) {
                double originalBiasGradient = currentLayer->bias_gradients->elements[outputNeuronIndex] = dLoss_dWeightedSums->elements[outputNeuronIndex];
                if(originalBiasGradient < network->optimization_config->gradient_clip_lower_bound) {
                    currentLayer->bias_gradients->elements[outputNeuronIndex] = network->optimization_config->gradient_clip_lower_bound;
                } else if(originalBiasGradient > network->optimization_config->gradient_clip_upper_bound) {
                    currentLayer->bias_gradients->elements[outputNeuronIndex] = network->optimization_config->gradient_clip_upper_bound;
                }
                double clippedBiasGradient = currentLayer->bias_gradients->elements[outputNeuronIndex];
                
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
            currentLayer->loss_wrt_wsums->elements[outputNeuronIndex] = dLoss_dWeightedSums->elements[outputNeuronIndex];
            #ifdef DEBUG
                log_debug( 
                    "Partial derivative of Loss with respect to the weighted sum for neuron #%d: %f \n", 
                    outputNeuronIndex, dLoss_dWeightedSums->elements[outputNeuronIndex]
                );
            #endif
        }

        #ifdef DEBUG
            char* gradients_str = matrix_to_string(currentLayer->gradients);
            char* bias_gradients_str = vector_to_string(currentLayer->biasGradients);
            log_debug(
                "Gradients for layer #%d: \n"
                "\t\t\t\tWeight gradients: \n%s\n"
                "\t\t\t\tBias gradients: %s\n",
                layerIndex, gradients_str, biasGradient_str
            );
            free(gradient_str);
            free(biasGradient_str);
        #endif
        
        free_matrix(jacobian);
        free_vector(dLoss_dOutputs);
        free_vector(dLoss_dWeightedSums);

        #ifdef DEBUG
            log_debug("Calculating gradients for the hidden layers of output index: %d", outputIndex);
        #endif
        for(layerIndex = network->num_layers - 2; layerIndex >= 0; layerIndex --) {
            currentLayer = network->layers[layerIndex];

            // l1 & l2 regularization
            Vector* l1_bias_derivatives;
            Vector* l2_bias_derivatives;

            if(currentLayer->bias_lambda > 0 ){
                l1_bias_derivatives = l1_derivative(currentLayer->bias_lambda, currentLayer->biases);
                l2_bias_derivatives = l2_derivative(currentLayer->bias_lambda, currentLayer->biases);
            }

            for(int neuronIndex = 0; neuronIndex < currentLayer->num_neurons; neuronIndex++) {
                double dLoss_dOutput = 0.0f;
                
                Layer* nextLayer = network->layers[layerIndex + 1];
                for(int neuronIndexNext = 0; neuronIndexNext < nextLayer->num_neurons; neuronIndexNext++) {
                    dLoss_dOutput += nextLayer->loss_wrt_wsums->elements[neuronIndexNext] * nextLayer->weights->data[neuronIndexNext]->elements[neuronIndex];
                }

                #ifdef DEBUG
                    log_debug("Partial derivative of Loss with respect to Output is: %f", dLoss_dOutput);
                #endif

                double dOutput_dWeightedSum = 0;
                
                switch(currentLayer->activation_fn) {
                    case RELU:
                        dOutput_dWeightedSum = relu_derivative(currentLayer->weighted_sums->elements[neuronIndex]);
                        break;
                    case LEAKY_RELU:
                        dOutput_dWeightedSum = leakyRelu_derivative(currentLayer->weighted_sums->elements[neuronIndex]);
                        break;
                    default:
                        break;
                }

                double dLoss_dWeightedSum = dLoss_dOutput * dOutput_dWeightedSum;
                #ifdef DEBUG
                    log_debug("Partial derivative of Output with respect to Net Input is: %f", dOutput_dWeightedSum);
                    log_debug("Partial derivative of Loss with respect to Net Input is: %f", dLoss_dWeightedSum);
                #endif
                Vector* l1_weights_derivatives;
                Vector* l2_weights_derivatives;

                if(currentLayer->weight_lambda > 0) {
                    l1_weights_derivatives = l1_derivative(currentLayer->weight_lambda, currentLayer->weights->data[neuronIndex]);
                    l2_weights_derivatives = l2_derivative(currentLayer->weight_lambda, currentLayer->weights->data[neuronIndex]);
                }

                for(int weightIndex = 0; weightIndex < currentLayer->weights->columns; weightIndex++) {
                    double dWeightedSum_dWeight = 0.0f;
                    if(layerIndex == 0) {
                        dWeightedSum_dWeight= input->elements[weightIndex];
                    }else {
                        dWeightedSum_dWeight = network->layers[layerIndex-1]->output->elements[weightIndex];
                    }
                    
                    
                    double dLoss_dWeight = dLoss_dWeightedSum * dWeightedSum_dWeight;
                    // if(currentLayer->weight_lambda > 0) {
                    //     dLoss_dWeight += l1_weights_derivatives->elements[weightIndex];
                    //     dLoss_dWeight += l2_weights_derivatives->elements[weightIndex];
                    // }

                    #ifdef DEBUG
                        log_debug(
                            "Partial derivative calculations for weight #%d of the neuron #%d: \n"
                            "\t\t\t\tdLoss_dWeightedSum: %f \n"
                            "\t\t\t\tdWeightedSum_dWeight: %f \n"
                            "\t\t\t\tdLoss_dWeight: %f \n", 
                            weightIndex, neuronIndex, dLoss_dWeightedSum, dWeightedSum_dWeight, dLoss_dWeight
                        );
                    #endif

                    currentLayer->weight_gradients->data[neuronIndex]->elements[weightIndex] += dLoss_dWeight;
                }
                
                currentLayer->bias_gradients->elements[neuronIndex] += dLoss_dWeightedSum;

                
                // if(currentLayer->bias_lambda > 0) {
                //     currentLayer->bias_gradients->elements[neuronIndex] += l1_bias_derivatives->elements[neuronIndex] + l2_bias_derivatives->elements[neuronIndex];
                // }

                if(network->optimization_config->use_gradient_clipping == 1) {
                    double originalBiasGradient = currentLayer->bias_gradients->elements[neuronIndex];
            
                    if (originalBiasGradient < network->optimization_config->gradient_clip_lower_bound) {
                        currentLayer->bias_gradients->elements[neuronIndex] = network->optimization_config->gradient_clip_lower_bound;
                    } else if (originalBiasGradient > network->optimization_config->gradient_clip_upper_bound) {
                        currentLayer->bias_gradients->elements[neuronIndex] = network->optimization_config->gradient_clip_upper_bound;
                    }
                    
                    double clippedBiasGradient = currentLayer->bias_gradients->elements[neuronIndex];

                    #ifdef DEBUG
                        char* gradients_str = matrix_to_string(currentLayer->gradients);
                        char* bias_gradients_str = vector_to_string(currentLayer->biasGradients);
                        log_debug(
                            "Gradients for layer #%d: \n"
                            "\t\t\t\tWeight gradients: \n%s\n"
                            "\t\t\t\tBias gradients: %s\n",
                            layerIndex, gradients_str, biasGradient_str
                        );
                        free(gradient_str);
                        free(biasGradient_str);
                    #endif
                }

                currentLayer->loss_wrt_wsums->elements[neuronIndex] = dLoss_dOutput * dOutput_dWeightedSum;

                // log_info("Gradient for weights and biases computed for neuron index: %d in layer index: %d", neuronIndex, layerIndex);
                if(currentLayer->weight_lambda > 0) {
                    free_vector(l1_weights_derivatives);
                    free_vector(l2_weights_derivatives);
                }
            }
            if(currentLayer->bias_lambda > 0) {
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