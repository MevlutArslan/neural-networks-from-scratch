#include "nnetwork.h"

void sgd(NNetwork* network, double learningRate, int batch_size) {
    double momentum = network->optimization_config->momentum;

    if(network->optimization_config->use_momentum == 1 && momentum == 0) {
        log_error("%s", "MOMENTUM HASN'T BEEN SET! \n");
        return;
    }

    for(int layer_index = 0; layer_index < network->num_layers; layer_index++) {
        Layer* current_layer = network->layers[layer_index];

        fill_matrix(current_layer->weight_momentums, 0.0f);
        fill_vector(current_layer->bias_momentums, 0.0f);
        
        for(int neuron_index = 0; neuron_index < current_layer->num_neurons; neuron_index++) {
            for(int weight_index = 0; weight_index < current_layer->weights->columns; weight_index++) {

                double weight_gradient = 0.0f;
                if(batch_size == 1) {
                    weight_gradient = network->weight_gradients[layer_index]->data[neuron_index]->elements[weight_index];
                }else if(batch_size == 0) {
                    weight_gradient = current_layer->weight_gradients->data[neuron_index]->elements[weight_index];
                }else {
                    log_error("not yet implemented!");
                    return;
                }

                double weight_update = learningRate * weight_gradient;
                    
                if(network->optimization_config->use_momentum == 1) {
                    double momentum_update = momentum * current_layer->weight_momentums->data[neuron_index]->elements[weight_index];
                    weight_update += momentum_update;

                    current_layer->weight_momentums->data[neuron_index]->elements[weight_index] = weight_update;
                }

                current_layer->weights->data[neuron_index]->elements[weight_index] -= weight_update;
            }

            double bias_gradient = 0.0f;
            if(batch_size == 1) {
                bias_gradient = network->bias_gradients[layer_index]->elements[neuron_index];
            }else if(batch_size == 0) {
                bias_gradient = current_layer->bias_gradients->elements[neuron_index];
            }else {
                log_error("not yet implemented!");
                return;
            } 

            double bias_update = learningRate * bias_gradient;

            if(network->optimization_config->use_momentum == 1) {
                double momentum_update = momentum * current_layer->bias_momentums->elements[neuron_index];
                bias_update += momentum_update;
                current_layer->bias_momentums->elements[neuron_index] = bias_update;
            }

            current_layer->biases->elements[neuron_index] -= bias_update;
        }
    }
}

void adagrad(NNetwork* network, double learningRate, int batch_size) {
    double epsilon = network->optimization_config->epsilon;

    if(epsilon == 0) {
        log_error("%s", "EPSILON HAS NOT BEEN SET! \n");
        return;
    }

    for(int layer_index = 0; layer_index < network->num_layers; layer_index++) {
        Layer* current_layer = network->layers[layer_index];

        fill_matrix(current_layer->weight_cache, 0.0f);
        fill_vector(current_layer->bias_cache, 0.0f);

        for(int neuron_index = 0; neuron_index < current_layer->num_neurons; neuron_index++) {
            for(int weight_index = 0; weight_index < current_layer->weights->columns; weight_index++) {
                double weight_gradient = 0.0f;
                if(batch_size == 1) {
                    weight_gradient = network->weight_gradients[layer_index]->data[neuron_index]->elements[weight_index];
                }else if(batch_size == 0) {
                    weight_gradient = current_layer->weight_gradients->data[neuron_index]->elements[weight_index];
                }else {
                    log_error("not yet implemented!");
                    return;
                }

                double weight_update = learningRate * weight_gradient;
                double gradient_squared = weight_gradient * weight_gradient;

                current_layer->weight_cache->data[neuron_index]->elements[weight_index] = gradient_squared;
                current_layer->weights->data[neuron_index]->elements[weight_index] -= weight_update / (sqrt(gradient_squared) + epsilon);
            }

            double bias_gradient = 0.0f;
            if(batch_size == 1) {
                bias_gradient = network->bias_gradients[layer_index]->elements[neuron_index];
            }else if(batch_size == 0) {
                bias_gradient = current_layer->bias_gradients->elements[neuron_index];
            }else {
                log_error("not yet implemented!");
                return;
            }

            double bias_update = learningRate * bias_gradient;
            double gradient_squared = bias_gradient * bias_gradient;

            current_layer->bias_cache->elements[neuron_index] = gradient_squared;
            current_layer->biases->elements[neuron_index] -= bias_update / (sqrt(gradient_squared) + epsilon);
        }
    }
}

void rms_prop(NNetwork* network, double learningRate, int batch_size) {
    double rho = network->optimization_config->rho;
    double epsilon = network->optimization_config->epsilon;

    if(rho == 0) {
        log_error("%s", "RHO HAS NOT BEEN SET! \n");
        return;
    }

    if(epsilon == 0) {
        log_error("%s", "EPSILON HAS NOT BEEN SET! \n");
        return;
    }

    for(int layer_index = 0; layer_index < network->num_layers; layer_index++) {
        Layer* current_layer = network->layers[layer_index];

        fill_matrix(current_layer->weight_cache, 0.0f);
        fill_vector(current_layer->bias_cache, 0.0f);

        for(int neuron_index = 0; neuron_index < current_layer->num_neurons; neuron_index++) {

            for(int weight_index = 0; weight_index < current_layer->weights->columns; weight_index++) {
                double weight_gradient = 0.0f;
                if(batch_size == 1) {
                    weight_gradient = network->weight_gradients[layer_index]->data[neuron_index]->elements[weight_index];
                }else if(batch_size == 0) {
                    weight_gradient = current_layer->weight_gradients->data[neuron_index]->elements[weight_index];
                }else {
                    log_error("not yet implemented!");
                    return;
                }
                double weight_update = -1 * (learningRate * weight_gradient);
                double w_gradient_squared = weight_gradient * weight_gradient;

                // cache = rho * cache + (1 - rho) * weight_gradient ** 2
                double cache_fraction = rho *  current_layer->weight_cache->data[neuron_index]->elements[weight_index];
                double gradient_squared_fraction = (1 - rho) * w_gradient_squared;

                current_layer->weight_cache->data[neuron_index]->elements[weight_index] = cache_fraction + gradient_squared_fraction;
                current_layer->weights->data[neuron_index]->elements[weight_index] += weight_update / (sqrt(cache_fraction + gradient_squared_fraction) + epsilon);
            }

            double bias_gradient = 0.0f;
            if(batch_size == 1) {
                bias_gradient = network->bias_gradients[layer_index]->elements[neuron_index];
            }else if(batch_size == 0) {
                bias_gradient = current_layer->bias_gradients->elements[neuron_index];
            }else {
                log_error("not yet implemented!");
                return;
            }

            double bias_update = -1 * (learningRate * bias_gradient);
            double gradient_squared = bias_gradient * bias_gradient;

            //  cache = rho * cache + (1 - rho) * gradient ** 2
            double cache_fraction = rho * current_layer->bias_cache->elements[neuron_index];
            double gradient_squared_fraction = (1 - rho) * gradient_squared;

            current_layer->bias_cache->elements[neuron_index] = cache_fraction + gradient_squared_fraction;
            current_layer->biases->elements[neuron_index] += bias_update / (sqrt(cache_fraction + gradient_squared_fraction) + epsilon);
        }
    }
}

void adam(NNetwork* network, double learningRate, int batch_size) {
    double beta1 = network->optimization_config->adam_beta1;
    double beta2 = network->optimization_config->adam_beta2;
    double epsilon = network->optimization_config->epsilon;

    if(beta1 == 0) {
        log_error("%s", "BETA_1 HAS NOT BEEN SET! \n");
        return;
    }

    if(beta2 == 0) {
        log_error("%s", "BETA_2 HAS NOT BEEN SET! \n");
        return;
    }

    if(epsilon == 0) {
        log_error("%s", "EPSILON HAS NOT BEEN SET! \n");
        return;
    }
    

    for(int layer_index = 0; layer_index < network->num_layers; layer_index++) {
        Layer* current_layer = network->layers[layer_index];

        fill_matrix(current_layer->weight_momentums, 0.0f);
        fill_vector(current_layer->bias_momentums, 0.0f);

        fill_matrix(current_layer->weight_cache, 0.0f);
        fill_vector(current_layer->bias_cache, 0.0f);

        for(int neuron_index = 0; neuron_index < current_layer->num_neurons; neuron_index++) {
            for(int weight_index = 0; weight_index < current_layer->weights->columns; weight_index++) {
                double weight_gradient = 0.0;
                if(batch_size == 1) {
                    weight_gradient = network->weight_gradients[layer_index]->data[neuron_index]->elements[weight_index];
                }else if(batch_size == 0) {
                    weight_gradient = network->layers[layer_index]->weight_gradients->data[neuron_index]->elements[weight_index];
                }else {
                    log_error("not yet implemented!");
                    return;
                }

                // m(t) = beta1 * m(t-1) + (1 – beta1) * g(t)
                double old_momentum =  current_layer->weight_momentums->data[neuron_index]->elements[weight_index];
                double weight_momentum = beta1 * old_momentum + (1 - beta1) * weight_gradient;

                current_layer->weight_momentums->data[neuron_index]->elements[weight_index] = weight_momentum;

                // mhat(t) = m(t) / (1 – beta1(t))
                double momentum_correction = weight_momentum / (1 - pow(beta1, network->training_epoch));

                #ifdef DEBUG
                    log_debug("Weight Momentum: %f \n"
                              "\t\t\t\tMomentum Correction: %f \n"
                              "\t\t\t\tBeta1: %f, Gradient: %f, Current Step: %d\n",
                              weightMomentum,
                              momentumCorrection,
                              beta1, gradient, network->currentStep);
                #endif

                // v(t) = beta2 * v(t-1) + (1 – beta2) * g(t)^2
                double old_cache = current_layer->weight_cache->data[neuron_index]->elements[weight_index];
                
                double weight_cache = beta2 * old_cache + (1 - beta2) * pow(weight_gradient, 2);
                current_layer->weight_cache->data[neuron_index]->elements[weight_index] = weight_cache;
                // vhat(t) = v(t) / (1 – beta2(t))
                double cache_correction = weight_cache / (1 - pow(beta2, network->training_epoch));

                #ifdef DEBUG
                    log_debug("Weight Cache: %f \n"
                              "\t\t\t\tCache Correction: %f \n"
                              "\t\t\t\tBeta2: %f, Gradient Squared: %f, Current Step: %d\n",
                              weightMomentum,
                              momentumCorrection,
                              beta1, pow(gradient, 2), network->currentStep);
                #endif
                // x(t) = x(t-1) – alpha * mhat(t) / (sqrt(vhat(t)) + eps)
                double old_weight = current_layer->weights->data[neuron_index]->elements[weight_index];
                double new_weight = old_weight - (learningRate * momentum_correction) / (sqrt(cache_correction) + epsilon);

                current_layer->weights->data[neuron_index]->elements[weight_index] = new_weight;
                #ifdef DEBUG
                    log_debug("Old Weight: %f \n"
                              "\t\t\t\tNew Weight %f \n",                     
                              old_weight,
                              currentLayer->weights->data[neuronIndex][weightIndex]);
                #endif
                // printf("%f, ", currentLayer->weights->data[neuronIndex]->elements[weightIndex]);

            }

            double bias_gradient = 0;
            if(batch_size == 1) {
                bias_gradient = network->bias_gradients[layer_index]->elements[neuron_index];
            }else if(batch_size == 0) {
                bias_gradient = network->layers[layer_index]->bias_gradients->elements[neuron_index];
            }else {
                log_error("not yet implemented!");
                return;
            }

            // BIAS UPDATE

            // Momentum calculations
            double bias_momentum = beta1 * current_layer->bias_momentums->elements[neuron_index] + (1 - beta1) * bias_gradient;
            current_layer->bias_momentums->elements[neuron_index] = bias_momentum;

            double momentum_correction = bias_momentum / (1 - pow(beta1, network->training_epoch));

            #ifdef DEBUG
                log_debug("Bias Momentum: %f \n"
                          "\t\t\t\tMomentum Correction: %f \n"
                          "\t\t\t\tBeta1: %f, Gradient: %f, Current Step: %d\n",
                          biasMomentum,
                          momentumCorrection,
                          beta1, biasGradient, network->currentStep);
            #endif
            // Cache calculations
            double bias_cache = beta2 * current_layer->bias_cache->elements[neuron_index] + (1 - beta2) * pow(bias_gradient, 2);
            current_layer->bias_cache->elements[neuron_index] = bias_cache;

            double cacheCorrection = bias_cache / (1 - pow(beta2, network->training_epoch));
            #ifdef DEBUG
                log_debug("Bias Cache: %f \n"
                          "\t\t\t\tCache Correction: %f \n"
                          "\t\t\t\tBeta2: %f, Gradient Squared: %f, Current Step: %d\n",
                          bias_cache,
                          cacheCorrection,
                          beta2, pow(biasGradient, 2), network->currentStep);
            #endif

            double old_bias = current_layer->biases->elements[neuron_index];
            double new_bias = old_bias - (learningRate * momentum_correction) / (sqrt(cacheCorrection) + epsilon);
            current_layer->biases->elements[neuron_index] = new_bias;
            
            #ifdef DEBUG
                log_debug("Old Weight: %f \n"
                          "\t\t\t\tNew Weight: %f \n",                     
                          oldBias,
                          current_layer->biases->elements[neuron_index]);
            #endif
        }
    }
}