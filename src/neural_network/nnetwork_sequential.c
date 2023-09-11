#include "nnetwork.h"


void forward_pass_sequential(NNetwork* network, Vector* input, Vector* output) {
    network->layers[0]->input = input;
    for (int layer_index = 0; layer_index < network->num_layers; layer_index++) {
        Layer* current_layer = network->layers[layer_index];
        Vector* product = dot_product(current_layer->weights, current_layer->input);
        
        #ifdef DEBUG
            char* weights_str = matrix_to_string(current_layer->weights);
            char* dot_product_str = vector_to_string(dotProduct);
            log_debug( 
                "Weights For Layer #%d: %s", 
                layer_index, 
                weights_str
            );

            log_debug( 
                "Dot Product For Layer #%d: %s", 
                layer_index, 
                dot_product_str
            );

            free(weights_str);
            free(dot_product_str);
        #endif
        
        vector_addition(product, current_layer->biases, current_layer->weighted_sums);
        
        #ifdef DEBUG
            char* weighted_sum_str = vector_to_string(currentLayer->output);
            log_debug(
                "Weighted sum for layer #%d: %s",  
                layer_index, 
                weighted_sum_str
            );
            free(weighted_sum_str);
        #endif

        current_layer->output = copy_vector(current_layer->weighted_sums);

        switch(current_layer->activation_fn) {
            case RELU:
                relu(current_layer->output);
                break;
            case LEAKY_RELU:
                leaky_relu(current_layer->output);
                break;
            case SOFTMAX:
                softmax(current_layer->output);
                break;
            default:
                log_error("Unknown Activation Function, be sure to register it to the workflow.");
                break;
        }

        #ifdef DEBUG
            char* output_str = vector_to_string(currentLayer->output);
            log_debug(
                "Output of activation function for layer #%d: %s", 
                layer_index, 
                output_str
            );
            free(output_str);
        #endif

        if(layer_index != network->num_layers - 1) {
            network->layers[layer_index + 1]->input = current_layer->output;
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
 * Performs the backpropagation process sequentially in the neural network.
 * 
 * @param network The neural network to perform backpropagation on.
 * @param input The input vector to the forward pass.
 * @param output The output of the forward pass.
 * @param target The vector with the expected outputs.
 */
void backpropagation_sequential(NNetwork* network, Vector* input, Vector* output, Vector* target) {
    #ifdef DEBUG
        log_info("Start of backward pass.");
    #endif
        Vector* prediction = output;
        // pre_declaring to get rid of compiler's errors
        // ------------------------------------------------------------------------
        Vector* loss_wrt_output;
        Matrix* output_wrt_weightedsum;

        double loss_wrt_output_mse;
        double output_wrt_wsum = 0.0f;
        double weighted_sum = 0.0f;
        // ------------------------------------------------------------------------

        Vector* loss_wrt_weightedsum = create_vector(network->layers[network->num_layers - 1]->num_neurons);
        switch(network->loss_fn) {
            case MEAN_SQUARED_ERROR:
                loss_wrt_output_mse = mean_squared_error_derivative(target->elements[0], output->elements[0]);

                switch(network->layers[network->num_layers - 1]->activation_fn) {
                    case RELU:
                        weighted_sum = network->layers[network->num_layers - 1]->weighted_sums->elements[0];
                        output_wrt_wsum = relu_derivative(weighted_sum);
                        break;
                    case LEAKY_RELU:
                        weighted_sum = network->layers[network->num_layers - 1]->weighted_sums->elements[0];
                        output_wrt_wsum = leaky_relu_derivative(weighted_sum);
                        break;
                    default:
                        log_error("Other activations functions haven't been implemented, feel free to add them!");
                        return;
                }

                loss_wrt_weightedsum->elements[0] = loss_wrt_output_mse * output_wrt_wsum;
                break;
            case CATEGORICAL_CROSS_ENTROPY:
                loss_wrt_output = categorical_cross_entropy_loss_derivative(target, prediction);

                output_wrt_weightedsum = softmax_derivative(prediction);
                
                dot_product_inplace(output_wrt_weightedsum, loss_wrt_output, loss_wrt_weightedsum);

                free_matrix(output_wrt_weightedsum);
                free_vector(loss_wrt_output);
                break;
            
            default:
                log_error("Unrecognized Loss Function, Please register your loss function!");
                return;
        }

        #ifdef DEBUG
            char* output_wrt_weightedsum_str = matrix_to_string(output_wrt_weightedsum);
            char* loss_wrt_weightedsum_str = vector_to_string(dLoss_dWeightedSums);
                
            log_debug("output wrt weightedsum is: %s", output_wrt_weightedsum_str);
            log_debug("loss wrt weightedsum is: %s", loss_wrt_weightedsum_str);
                
            free(output_wrt_weightedsum_str);
            free(loss_wrt_weightedsum_str);
        #endif

        int layer_index = network->num_layers - 1;
        Layer* current_layer = network->layers[layer_index];
        for(int neuron_index = 0; neuron_index < current_layer->num_neurons; neuron_index++) {
            #ifdef DEBUG
                log_debug("Partial derivative of Loss with respect to Output for target: %f, prediction: %f is %f \n", target->elements[neuron_index], prediction->elements[neuron_index], dLoss_dOutputs->elements[neuron_index]);
            #endif

            #ifdef DEBUG
                log_debug("Partial derivative of Loss with respect to the input to the activation function of the output layer's node #%d is: %f \n", neuron_index ,dLoss_dWeightedSums->elements[neuron_index]);
            #endif
            
            for(int weight_index = 0; weight_index < current_layer->weights->columns; weight_index++) {
                double weightedsum_wrt_weight = 0.0f;
                if(layer_index == 0) {
                    weightedsum_wrt_weight= input->elements[weight_index];
                }else {
                    weightedsum_wrt_weight = network->layers[layer_index - 1]->output->elements[weight_index];
                }

                double loss_wrt_weight = loss_wrt_weightedsum->elements[neuron_index] * weightedsum_wrt_weight;

                #ifdef DEBUG
                    log_debug( 
                        "Partial derivative calculations for weight #%d of the neuron #%d: \n"
                        "\t\t\t\t loss wrt weightedsum: %f \n"
                        "\t\t\t\t weighedsum wrt weight: %f \n"
                        "\t\t\t\t loss wrt weight: %f \n", 
                        weight_index, neuron_index, loss_wrt_weightedsum->elements[neuron_index], weightedsum_wrt_weight, loss_wrt_weight
                    );
                #endif

                current_layer->weight_gradients->data[neuron_index]->elements[weight_index] += loss_wrt_weight;
                
            }

            current_layer->bias_gradients->elements[neuron_index] += loss_wrt_weightedsum->elements[neuron_index];
  
            // Backpropagating the error to the hidden layers
            current_layer->loss_wrt_wsums->elements[neuron_index] = loss_wrt_weightedsum->elements[neuron_index];
            #ifdef DEBUG
                log_debug( 
                    "Partial derivative of Loss with respect to the weighted sum for neuron #%d: %f \n", 
                    neuron_index, loss_wrt_weightedsum->elements[neuron_index]
                );
            #endif
        }

        #ifdef DEBUG
            char* weight_gradients_str = matrix_to_string(currentLayer->weight_gradients);
            char* bias_gradient_str = vector_to_string(currentLayer->bias_gradients);
            log_debug(
                "Gradients for layer #%d: \n"
                "\t\t\t\tWeight gradients: \n%s\n"
                "\t\t\t\tBias gradients: %s\n",
                layer_index, weight_gradients_str, bias_gradient_str
            );
            free(weight_gradients_str);
            free(bias_gradient_str);
        #endif
        
        // clean up memory used during output layer's backpropagation step.
        free_vector(loss_wrt_weightedsum);

        #ifdef DEBUG
            log_debug("Calculating gradients for the hidden layers of output index: %d", neuron_index);
        #endif
        for(layer_index = network->num_layers - 2; layer_index >= 0; layer_index --) {
            current_layer = network->layers[layer_index];

            // l1 &/or l2 regularization
            Vector* l1_bias_derivatives;
            Vector* l2_bias_derivatives;
            
            if(current_layer->l1_bias_lambda > 0) {
                l1_bias_derivatives = l1_derivative(current_layer->l1_bias_lambda, current_layer->biases);
            }
            
            if(current_layer->l2_bias_lambda > 0) {
                l2_bias_derivatives = l2_derivative(current_layer->l2_bias_lambda, current_layer->biases);
            }

            for(int neuron_index = 0; neuron_index < current_layer->num_neurons; neuron_index++) {
                double loss_wrt_output = 0.0f;
                
                Layer* nextLayer = network->layers[layer_index + 1];
                for(int neuron_index_next_layer = 0; neuron_index_next_layer < nextLayer->num_neurons; neuron_index_next_layer++) {
                    loss_wrt_output += nextLayer->loss_wrt_wsums->elements[neuron_index_next_layer] * nextLayer->weights->data[neuron_index_next_layer]->elements[neuron_index];
                }

                #ifdef DEBUG
                    log_debug("Partial derivative of Loss with respect to Output is: %f", loss_wrt_output);
                #endif

                double output_wrt_weightedsum = 0;
                
                switch(current_layer->activation_fn) {
                    case RELU:
                        output_wrt_weightedsum = relu_derivative(current_layer->weighted_sums->elements[neuron_index]);
                        break;
                    case LEAKY_RELU:
                        output_wrt_weightedsum = leaky_relu_derivative(current_layer->weighted_sums->elements[neuron_index]);
                        break;
                    default:
                        break;
                }

                double loss_wrt_weightedsum = loss_wrt_output * output_wrt_weightedsum;
                #ifdef DEBUG
                    log_debug("Partial derivative of Output with respect to weighted sum is: %f", output_wrt_weightedsum);
                    log_debug("Partial derivative of Loss with respect to Net Input is: %f", loss_wrt_weightedsum);
                #endif
                // Vector* l1_weights_derivatives;
                // Vector* l2_weights_derivatives;

                // if(current_layer->l1_weight_lambda > 0) {
                //     l1_weights_derivatives = l1_derivative(current_layer->l1_weight_lambda, current_layer->biases);
                // }
            
                // if(current_layer->l2_weight_lambda > 0) {
                //     l2_weights_derivatives = l2_derivative(current_layer->l2_weight_lambda, current_layer->biases);
                // }

                for(int weightIndex = 0; weightIndex < current_layer->weights->columns; weightIndex++) {
                    double weightedsum_wrt_weight = 0.0f;
                    if(layer_index == 0) {
                        weightedsum_wrt_weight= input->elements[weightIndex];
                    }else {
                        weightedsum_wrt_weight = network->layers[layer_index-1]->output->elements[weightIndex];
                    }
                    
                    
                    double loss_wrt_weight = loss_wrt_weightedsum * weightedsum_wrt_weight;
                    // if(current_layer->l1_weight_lambda > 0) {
                    //     dLoss_dWeight += l1_weights_derivatives->elements[weightIndex];
                    // }

                    // if(current_layer->l2_weight_lambda > 0) {
                    //     dLoss_dWeight += l2_weights_derivatives->elements[weightIndex];
                    // }

                    #ifdef DEBUG
                        log_debug(
                            "Partial derivative calculations for weight #%d of the neuron #%d: \n"
                            "\t\t\t\tdLoss_dWeightedSum: %f \n"
                            "\t\t\t\tdWeightedSum_dWeight: %f \n"
                            "\t\t\t\tdLoss_dWeight: %f \n", 
                            weightIndex, neuronIndex, loss_wrt_weightedsum, weightedsum_wrt_weight, loss_wrt_weight
                        );
                    #endif

                    current_layer->weight_gradients->data[neuron_index]->elements[weightIndex] += loss_wrt_weight;
                }
                
                current_layer->bias_gradients->elements[neuron_index] += loss_wrt_weightedsum;

                if(current_layer->l1_bias_lambda > 0) {
                    current_layer->bias_gradients->elements[neuron_index] += l1_bias_derivatives->elements[neuron_index];
                }

                if(current_layer->l2_bias_lambda > 0) {
                    current_layer->bias_gradients->elements[neuron_index] += l2_bias_derivatives->elements[neuron_index];
                }

                current_layer->loss_wrt_wsums->elements[neuron_index] = loss_wrt_output * output_wrt_weightedsum;

                // log_info("Gradient for weights and biases computed for neuron index: %d in layer index: %d", neuronIndex, layerIndex);
                // if(current_layer->l1_weight_lambda > 0) {
                //     free_vector(l1_weights_derivatives);
                // }

                // if(current_layer->l2_weight_lambda > 0) {
                //     free_vector(l2_weights_derivatives);
                // }
            }

            // if(current_layer->l1_bias_lambda > 0) {
            //     free_vector(l1_bias_derivatives);
            // }

            // if(current_layer->l2_bias_lambda > 0) {
            //     free_vector(l2_bias_derivatives);
            // }
           
            #ifdef DEBUG
                char* weight_gradient_str = matrix_to_string(current_layer->weight_gradients);
                char* bias_gradient_str = vector_to_string(current_layer->bias_gradient);
                log_debug( 
                    "Gradients for layer #%d: \n"
                    "\t\t\t\tWeight gradients: \n%s\n"
                    "\t\t\t\tBias gradients: %s\n",
                    layer_index, matrix_to_string(currentLayer->weight_gradients), vector_to_string(currentLayer->bias_gradient)
                );
            #endif
        }    
    #ifdef DEBUG
        log_info("End of backward pass.");
    #endif
}