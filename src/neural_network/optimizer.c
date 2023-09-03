#include "nnetwork.h"

void sgd(NNetwork* network, double learningRate, int batch_size) {
    double momentum = network->optimization_config->momentum;

    if(network->optimization_config->use_momentum == 1 && momentum == 0) {
        log_error("%s", "MOMENTUM HASN'T BEEN SET! \n");
        return;
    }

    Matrix** weight_gradients = network->weight_gradients;
    Vector** bias_gradients = network->bias_gradients;

    for(int layerIndex = 0; layerIndex < network->num_layers; layerIndex++) {
        Layer* currentLayer = network->layers[layerIndex];
        fill_matrix(currentLayer->weight_momentums, 0.0f);
        fill_vector(currentLayer->bias_momentums, 0.0f);
        
        for(int neuronIndex = 0; neuronIndex < currentLayer->num_neurons; neuronIndex++) {
            for(int weightIndex = 0; weightIndex < currentLayer->weights->columns; weightIndex++) {
                double gradient = weight_gradients[layerIndex]->data[neuronIndex]->elements[weightIndex];
                double valueToUpdateBy = learningRate * gradient;
                    
                if(network->optimization_config->use_momentum == 1) {
                    double momentumUpdate = momentum * currentLayer->weight_momentums->data[neuronIndex]->elements[weightIndex];
                    valueToUpdateBy += momentumUpdate;

                    currentLayer->weight_momentums->data[neuronIndex]->elements[weightIndex] = valueToUpdateBy;
                }

                currentLayer->weights->data[neuronIndex]->elements[weightIndex] -= valueToUpdateBy;
            }
            double biasGradient = bias_gradients[layerIndex]->elements[neuronIndex];
            double biasValueToUpdateBy = learningRate * biasGradient;

            if(network->optimization_config->use_momentum == 1) {
                double momentumUpdate = momentum * currentLayer->bias_momentums->elements[neuronIndex];
                biasValueToUpdateBy += momentumUpdate;
                currentLayer->bias_momentums->elements[neuronIndex] = biasValueToUpdateBy;
            }

            currentLayer->biases->elements[neuronIndex] -= biasValueToUpdateBy;
        }
    }
}

void adagrad(NNetwork* network, double learningRate, int batch_size) {
    double epsilon = network->optimization_config->epsilon;

    if(epsilon == 0) {
        log_error("%s", "EPSILON HAS NOT BEEN SET! \n");
        return;
    }

    Matrix** weight_gradients = network->weight_gradients;
    Vector** bias_gradients = network->bias_gradients;

    
    for(int layerIndex = 0; layerIndex < network->num_layers; layerIndex++) {
        Layer* currentLayer = network->layers[layerIndex];
        fill_matrix(currentLayer->weight_cache, 0.0f);
        fill_vector(currentLayer->bias_cache, 0.0f);
        for(int neuronIndex = 0; neuronIndex < currentLayer->num_neurons; neuronIndex++) {
            double biasGradient = bias_gradients[layerIndex]->elements[neuronIndex];
            // TODO rename biasValueToUpdateBy!!!!
            double biasValueToUpdateBy = learningRate * biasGradient;
            double biasGradientSquared = biasGradient * biasGradient;
            
            for(int weightIndex = 0; weightIndex < currentLayer->weights->columns; weightIndex++) {
                double gradient = weight_gradients[layerIndex]->data[neuronIndex]->elements[weightIndex];
                // TODO rename valueToUpdateBy!!!!
                double valueToUpdateBy = learningRate * gradient;
                double gradientSquared = gradient * gradient;

                currentLayer->weight_cache->data[neuronIndex]->elements[weightIndex] = gradientSquared;
                currentLayer->weights->data[neuronIndex]->elements[weightIndex] -= valueToUpdateBy / (sqrt(gradientSquared) + epsilon);
            }

            currentLayer->bias_cache->elements[neuronIndex] = biasGradientSquared;
            currentLayer->biases->elements[neuronIndex] -= biasValueToUpdateBy / (sqrt(biasGradientSquared) + epsilon);
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
    
    Matrix** weight_gradients = network->weight_gradients;
    Vector** bias_gradients = network->bias_gradients;

    for(int layerIndex = 0; layerIndex < network->num_layers; layerIndex++) {
        Layer* currentLayer = network->layers[layerIndex];
        fill_matrix(currentLayer->weight_cache, 0.0f);
        fill_vector(currentLayer->bias_cache, 0.0f);
        for(int neuronIndex = 0; neuronIndex < currentLayer->num_neurons; neuronIndex++) {
            double biasGradient = bias_gradients[layerIndex]->elements[neuronIndex];
            // TODO rename biasValueToUpdateBy!!!!
            double biasValueToUpdateBy = -1 * (learningRate * biasGradient);
            double biasGradientSquared = biasGradient * biasGradient;

            for(int weightIndex = 0; weightIndex < currentLayer->weights->columns; weightIndex++) {
                double gradient = weight_gradients[layerIndex]->data[neuronIndex]->elements[weightIndex];
                // TODO rename valueToUpdateBy!!!!
                double valueToUpdateBy = -1 * (learningRate * gradient);
                double gradientSquared = gradient * gradient;

                // cache = rho * cache + (1 - rho) * gradient ** 2
                double fractionOfCache = rho *  currentLayer->weight_cache->data[neuronIndex]->elements[weightIndex];
                double fractionOfGradientSquarred = (1 - rho) * gradientSquared;

                currentLayer->weight_cache->data[neuronIndex]->elements[weightIndex] = fractionOfCache + fractionOfGradientSquarred;
                currentLayer->weights->data[neuronIndex]->elements[weightIndex] += valueToUpdateBy / (sqrt(fractionOfCache + fractionOfGradientSquarred) + epsilon);
            }

            //  cache = rho * cache + (1 - rho) * gradient ** 2
            double fractionOfCache = rho * currentLayer->bias_cache->elements[neuronIndex];
            double fractionOfGradientSquarred = (1 - rho) * biasGradientSquared;
            currentLayer->bias_cache->elements[neuronIndex] = fractionOfCache + fractionOfGradientSquarred;
            currentLayer->biases->elements[neuronIndex] += biasValueToUpdateBy / (sqrt(fractionOfCache + fractionOfGradientSquarred) + epsilon);
        }
    }
}

void adam(NNetwork* network, double learningRate, int batch_size) {
    double beta1 = network->optimization_config->adam_beta1;
    double beta2 = network->optimization_config->adam_beta2;
    double epsilon = network->optimization_config->epsilon;

    Matrix** weight_gradients = network->weight_gradients;
    Vector** bias_gradients = network->bias_gradients;

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
    

    for(int layerIndex = 0; layerIndex < network->num_layers; layerIndex++) {
        Layer* currentLayer = network->layers[layerIndex];

        fill_matrix(currentLayer->weight_momentums, 0.0f);
        fill_vector(currentLayer->bias_momentums, 0.0f);

        fill_matrix(currentLayer->weight_cache, 0.0f);
        fill_vector(currentLayer->bias_cache, 0.0f);

        for(int neuronIndex = 0; neuronIndex < currentLayer->num_neurons; neuronIndex++) {
            for(int weightIndex = 0; weightIndex < currentLayer->weights->columns; weightIndex++) {
                double weight_gradient = 0.0;
                if(batch_size == 1) {
                    weight_gradient = weight_gradients[layerIndex]->data[neuronIndex]->elements[weightIndex];
                }else if(batch_size == 0) {
                    weight_gradient = network->layers[layerIndex]->weight_gradients->data[neuronIndex]->elements[weightIndex];
                }
                // m(t) = beta1 * m(t-1) + (1 – beta1) * g(t)
                double old_momentum =  currentLayer->weight_momentums->data[neuronIndex]->elements[weightIndex];
                double weightMomentum = beta1 * old_momentum + (1 - beta1) * weight_gradient;

                currentLayer->weight_momentums->data[neuronIndex]->elements[weightIndex] = weightMomentum;
                // mhat(t) = m(t) / (1 – beta1(t))
                double momentumCorrection = weightMomentum / (1 - pow(beta1, network->training_epoch));

                #ifdef DEBUG
                    log_debug("Weight Momentum: %f \n"
                              "\t\t\t\tMomentum Correction: %f \n"
                              "\t\t\t\tBeta1: %f, Gradient: %f, Current Step: %d\n",
                              weightMomentum,
                              momentumCorrection,
                              beta1, gradient, network->currentStep);
                #endif

                // v(t) = beta2 * v(t-1) + (1 – beta2) * g(t)^2
                double old_cache = currentLayer->weight_cache->data[neuronIndex]->elements[weightIndex];
                double weight_cache = beta2 * old_cache + (1 - beta2) * pow(weight_gradient, 2);
                currentLayer->weight_cache->data[neuronIndex]->elements[weightIndex] = weight_cache;

                // vhat(t) = v(t) / (1 – beta2(t))
                double cacheCorrection = weight_cache / (1 - pow(beta2, network->training_epoch));

                #ifdef DEBUG
                    log_debug("Weight Cache: %f \n"
                              "\t\t\t\tCache Correction: %f \n"
                              "\t\t\t\tBeta2: %f, Gradient Squared: %f, Current Step: %d\n",
                              weightMomentum,
                              momentumCorrection,
                              beta1, pow(gradient, 2), network->currentStep);
                #endif
                // x(t) = x(t-1) – alpha * mhat(t) / (sqrt(vhat(t)) + eps)
                double oldWeight = currentLayer->weights->data[neuronIndex]->elements[weightIndex];
                double newWeight = oldWeight - (learningRate * momentumCorrection) / (sqrt(cacheCorrection) + epsilon);

                currentLayer->weights->data[neuronIndex]->elements[weightIndex] = newWeight;
                #ifdef DEBUG
                    log_debug("Old Weight: %f \n"
                              "\t\t\t\tNew Weight %f \n",                     
                              oldWeight,
                              currentLayer->weights->data[neuronIndex][weightIndex]);
                #endif
            }

            double biasGradient = 0;
            if(batch_size == 1) {
                biasGradient = bias_gradients[layerIndex]->elements[neuronIndex];
            }else if(batch_size == 0) {
                biasGradient = network->layers[layerIndex]->bias_gradients->elements[neuronIndex];
            }

            // BIAS UPDATE

            // Momentum calculations
            double biasMomentum = beta1 * currentLayer->bias_momentums->elements[neuronIndex] + (1 - beta1) * biasGradient;
            currentLayer->bias_momentums->elements[neuronIndex] = biasMomentum;

            double momentumCorrection = biasMomentum / (1 - pow(beta1, network->training_epoch));

            #ifdef DEBUG
                log_debug("Bias Momentum: %f \n"
                          "\t\t\t\tMomentum Correction: %f \n"
                          "\t\t\t\tBeta1: %f, Gradient: %f, Current Step: %d\n",
                          biasMomentum,
                          momentumCorrection,
                          beta1, biasGradient, network->currentStep);
            #endif
            // Cache calculations
            double bias_cache = beta2 * currentLayer->bias_cache->elements[neuronIndex] + (1 - beta2) * pow(biasGradient, 2);
            currentLayer->bias_cache->elements[neuronIndex] = bias_cache;

            double cacheCorrection = bias_cache / (1 - pow(beta2, network->training_epoch));
            #ifdef DEBUG
                log_debug("Bias Cache: %f \n"
                          "\t\t\t\tCache Correction: %f \n"
                          "\t\t\t\tBeta2: %f, Gradient Squared: %f, Current Step: %d\n",
                          bias_cache,
                          cacheCorrection,
                          beta2, pow(biasGradient, 2), network->currentStep);
            #endif

            double oldBias = currentLayer->biases->elements[neuronIndex];
            currentLayer->biases->elements[neuronIndex] -= (learningRate * momentumCorrection) / (sqrt(cacheCorrection) + epsilon);

            #ifdef DEBUG
                log_debug("Old Weight: %f \n"
                          "\t\t\t\tNew Weight: %f \n",                     
                          oldBias,
                          currentLayer->biases->elements[neuronIndex]);
            #endif
        }
    }
}