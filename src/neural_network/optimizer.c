#include "nnetwork.h"

void sgd(NNetwork* network, double learningRate) {
    double momentum = network->optimizationConfig->momentum;

    if(network->optimizationConfig->shouldUseMomentum == 1 && momentum == 0) {
        log_error("%s", "MOMENTUM HASN'T BEEN SET! \n");
        return;
    }

    Matrix** weight_gradients = network->weight_gradients;
    Vector** bias_gradients = network->bias_gradients;


    for(int layerIndex = 0; layerIndex < network->layerCount; layerIndex++) {
        Layer* currentLayer = network->layers[layerIndex];
        fill_matrix(currentLayer->weightMomentums, 0.0f);
        fill_vector(currentLayer->biasMomentums, 0.0f);
        
        for(int neuronIndex = 0; neuronIndex < currentLayer->neuronCount; neuronIndex++) {
            for(int weightIndex = 0; weightIndex < currentLayer->weights->columns; weightIndex++) {
                double weight_gradient = weight_gradients[layerIndex]->get_element(weight_gradients[layerIndex], neuronIndex, weightIndex);
                double valueToUpdateBy = learningRate * weight_gradient;
                    
                if(network->optimizationConfig->shouldUseMomentum == 1) {
                    double momentumUpdate = momentum * currentLayer->weightMomentums->get_element(currentLayer->weightMomentums, neuronIndex, weightIndex);
                    valueToUpdateBy += momentumUpdate;

                    currentLayer->weightMomentums->set_element(currentLayer->weightMomentums, neuronIndex, weightIndex, valueToUpdateBy);
                }
                
                double old_weight = currentLayer->weights->get_element(currentLayer->weights, neuronIndex, weightIndex);
                double new_weight = old_weight - valueToUpdateBy;

                currentLayer->weights->set_element(currentLayer->weights, neuronIndex, weightIndex, new_weight);
            }
            double biasGradient = bias_gradients[layerIndex]->elements[neuronIndex];
            double biasValueToUpdateBy = learningRate * biasGradient;

            if(network->optimizationConfig->shouldUseMomentum == 1) {
                double momentumUpdate = momentum * currentLayer->biasMomentums->elements[neuronIndex];
                biasValueToUpdateBy += momentumUpdate;
                currentLayer->biasMomentums->elements[neuronIndex] = biasValueToUpdateBy;
            }

            currentLayer->biases->elements[neuronIndex] -= biasValueToUpdateBy;
        }
    }
}

void adagrad(NNetwork* network, double learningRate) {
    double epsilon = network->optimizationConfig->epsilon;

    if(epsilon == 0) {
        log_error("%s", "EPSILON HAS NOT BEEN SET! \n");
        return;
    }

    Matrix** weight_gradients = network->weight_gradients;
    Vector** bias_gradients = network->bias_gradients;

    
    for(int layerIndex = 0; layerIndex < network->layerCount; layerIndex++) {
        Layer* currentLayer = network->layers[layerIndex];
        fill_matrix(currentLayer->weightCache, 0.0f);
        fill_vector(currentLayer->biasCache, 0.0f);
        for(int neuronIndex = 0; neuronIndex < currentLayer->neuronCount; neuronIndex++) {
            double biasGradient = bias_gradients[layerIndex]->elements[neuronIndex];
            // TODO rename biasValueToUpdateBy!!!!
            double biasValueToUpdateBy = learningRate * biasGradient;
            double biasGradientSquared = biasGradient * biasGradient;
            
            for(int weightIndex = 0; weightIndex < currentLayer->weights->columns; weightIndex++) {
                double weight_gradient = weight_gradients[layerIndex]->get_element(weight_gradients[layerIndex], neuronIndex, weightIndex);
                // TODO rename valueToUpdateBy!!!!
                double valueToUpdateBy = learningRate * weight_gradient;
                double gradientSquared = weight_gradient * weight_gradient;
                
                currentLayer->weightCache->set_element(currentLayer->weightCache, neuronIndex, weightIndex, gradientSquared);

                double old_weight = currentLayer->weights->get_element(currentLayer->weights, neuronIndex, weightIndex);
                double new_weight = old_weight - valueToUpdateBy / (sqrt(gradientSquared) + epsilon);
                currentLayer->weights->set_element(currentLayer->weights, neuronIndex, weightIndex, new_weight);
            }

            currentLayer->biasCache->elements[neuronIndex] = biasGradientSquared;
            currentLayer->biases->elements[neuronIndex] -= biasValueToUpdateBy / (sqrt(biasGradientSquared) + epsilon);
        }
    }
}

void rms_prop(NNetwork* network, double learningRate) {
    double rho = network->optimizationConfig->rho;
    double epsilon = network->optimizationConfig->epsilon;

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

    for(int layerIndex = 0; layerIndex < network->layerCount; layerIndex++) {
        Layer* currentLayer = network->layers[layerIndex];
        fill_matrix(currentLayer->weightCache, 0.0f);
        fill_vector(currentLayer->biasCache, 0.0f);
        for(int neuronIndex = 0; neuronIndex < currentLayer->neuronCount; neuronIndex++) {
            double biasGradient = bias_gradients[layerIndex]->elements[neuronIndex];
            // TODO rename biasValueToUpdateBy!!!!
            double biasValueToUpdateBy = -1 * (learningRate * biasGradient);
            double biasGradientSquared = biasGradient * biasGradient;

            for(int weightIndex = 0; weightIndex < currentLayer->weights->columns; weightIndex++) {
                double weight_gradient = weight_gradients[layerIndex]->get_element(weight_gradients[layerIndex], neuronIndex, weightIndex);
                // TODO rename valueToUpdateBy!!!!
                double valueToUpdateBy = -1 * (learningRate * weight_gradient);
                double gradientSquared = weight_gradient * weight_gradient;

                // cache = rho * cache + (1 - rho) * gradient ** 2
                double fractionOfCache = rho *  currentLayer->weightCache->get_element(currentLayer->weightCache, neuronIndex, weightIndex);
                double fractionOfGradientSquarred = (1 - rho) * gradientSquared;

                currentLayer->weightCache->set_element(currentLayer->weightCache, neuronIndex, weightIndex, fractionOfCache + fractionOfGradientSquarred);
                double old_weight = currentLayer->weights->get_element(currentLayer->weights, neuronIndex, weightIndex);
                double new_weight = old_weight += valueToUpdateBy / (sqrt(fractionOfCache + fractionOfGradientSquarred) + epsilon);
                currentLayer->weights->set_element(currentLayer->weights, neuronIndex, weightIndex, new_weight);
            }

            //  cache = rho * cache + (1 - rho) * gradient ** 2
            double fractionOfCache = rho * currentLayer->biasCache->elements[neuronIndex];
            double fractionOfGradientSquarred = (1 - rho) * biasGradientSquared;
            currentLayer->biasCache->elements[neuronIndex] = fractionOfCache + fractionOfGradientSquarred;
            currentLayer->biases->elements[neuronIndex] += biasValueToUpdateBy / (sqrt(fractionOfCache + fractionOfGradientSquarred) + epsilon);
        }
    }
}

void adam(NNetwork* network, double learningRate) {
    double beta1 = network->optimizationConfig->beta1;
    double beta2 = network->optimizationConfig->beta2;
    double epsilon = network->optimizationConfig->epsilon;

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
    

    for(int layerIndex = 0; layerIndex < network->layerCount; layerIndex++) {
        Layer* currentLayer = network->layers[layerIndex];

        fill_matrix(currentLayer->weightMomentums, 0.0f);
        fill_vector(currentLayer->biasMomentums, 0.0f);

        fill_matrix(currentLayer->weightCache, 0.0f);
        fill_vector(currentLayer->biasCache, 0.0f);
        for(int neuronIndex = 0; neuronIndex < currentLayer->neuronCount; neuronIndex++) {
            for(int weightIndex = 0; weightIndex < currentLayer->weights->columns; weightIndex++) {
                double weight_gradient = weight_gradients[layerIndex]->get_element(weight_gradients[layerIndex], neuronIndex, weightIndex);

                // m(t) = beta1 * m(t-1) + (1 – beta1) * g(t)
                double old_momentum =  currentLayer->weightMomentums->get_element(currentLayer->weightMomentums, neuronIndex, weightIndex);
                double weightMomentum = beta1 * old_momentum + (1 - beta1) * weight_gradient;
                currentLayer->weightMomentums->set_element(currentLayer->weightMomentums, neuronIndex, weightIndex, weightMomentum);
                // mhat(t) = m(t) / (1 – beta1(t))
                double momentumCorrection = weightMomentum / (1 - pow(beta1, network->currentStep));

                #ifdef DEBUG
                    log_debug("Weight Momentum: %f \n"
                              "\t\t\t\tMomentum Correction: %f \n"
                              "\t\t\t\tBeta1: %f, Gradient: %f, Current Step: %d\n",
                              weightMomentum,
                              momentumCorrection,
                              beta1, gradient, network->currentStep);
                #endif

                // v(t) = beta2 * v(t-1) + (1 – beta2) * g(t)^2
                double old_cache = currentLayer->weightCache->get_element(currentLayer->weightCache, neuronIndex, weightIndex);
                double weightCache = beta2 * old_cache + (1 - beta2) * pow(weight_gradient, 2);
                currentLayer->weightCache->set_element(currentLayer->weightCache, neuronIndex, weightIndex, weightCache);

                // vhat(t) = v(t) / (1 – beta2(t))
                double cacheCorrection = weightCache / (1 - pow(beta2, network->currentStep));

                #ifdef DEBUG
                    log_debug("Weight Cache: %f \n"
                              "\t\t\t\tCache Correction: %f \n"
                              "\t\t\t\tBeta2: %f, Gradient Squared: %f, Current Step: %d\n",
                              weightMomentum,
                              momentumCorrection,
                              beta1, pow(gradient, 2), network->currentStep);
                #endif
                // x(t) = x(t-1) – alpha * mhat(t) / (sqrt(vhat(t)) + eps)
                double oldWeight = currentLayer->weights->get_element(currentLayer->weights, neuronIndex, weightIndex);
                double newWeight = oldWeight - (learningRate * momentumCorrection) / (sqrt(cacheCorrection) + epsilon);
                currentLayer->weights->set_element(currentLayer->weights, neuronIndex, weightIndex, newWeight);
                #ifdef DEBUG
                    log_debug("Old Weight: %f \n"
                              "\t\t\t\tNew Weight %f \n",                     
                              oldWeight,
                              currentLayer->weights->data[neuronIndex][weightIndex]);
                #endif
            }

            double biasGradient = bias_gradients[layerIndex]->elements[neuronIndex];

            // BIAS UPDATE

            // Momentum calculations
            double biasMomentum = beta1 * currentLayer->biasMomentums->elements[neuronIndex] + (1 - beta1) * biasGradient;
            currentLayer->biasMomentums->elements[neuronIndex] = biasMomentum;

            double momentumCorrection = biasMomentum / (1 - pow(beta1, network->currentStep));

            #ifdef DEBUG
                log_debug("Bias Momentum: %f \n"
                          "\t\t\t\tMomentum Correction: %f \n"
                          "\t\t\t\tBeta1: %f, Gradient: %f, Current Step: %d\n",
                          biasMomentum,
                          momentumCorrection,
                          beta1, biasGradient, network->currentStep);
            #endif
            // Cache calculations
            double biasCache = beta2 * currentLayer->biasCache->elements[neuronIndex] + (1 - beta2) * pow(biasGradient, 2);
            currentLayer->biasCache->elements[neuronIndex] = biasCache;

            double cacheCorrection = biasCache / (1 - pow(beta2, network->currentStep));
            #ifdef DEBUG
                log_debug("Bias Cache: %f \n"
                          "\t\t\t\tCache Correction: %f \n"
                          "\t\t\t\tBeta2: %f, Gradient Squared: %f, Current Step: %d\n",
                          biasCache,
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
