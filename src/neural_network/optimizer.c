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
                double gradient = weight_gradients[layerIndex]->data[neuronIndex]->elements[weightIndex];
                double valueToUpdateBy = learningRate * gradient;
                    
                if(network->optimizationConfig->shouldUseMomentum == 1) {
                    double momentumUpdate = momentum * currentLayer->weightMomentums->data[neuronIndex]->elements[weightIndex];
                    valueToUpdateBy += momentumUpdate;

                    currentLayer->weightMomentums->data[neuronIndex]->elements[weightIndex] = valueToUpdateBy;
                }

                currentLayer->weights->data[neuronIndex]->elements[weightIndex] -= valueToUpdateBy;
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
                double gradient = weight_gradients[layerIndex]->data[neuronIndex]->elements[weightIndex];
                // TODO rename valueToUpdateBy!!!!
                double valueToUpdateBy = learningRate * gradient;
                double gradientSquared = gradient * gradient;

                currentLayer->weightCache->data[neuronIndex]->elements[weightIndex] = gradientSquared;
                currentLayer->weights->data[neuronIndex]->elements[weightIndex] -= valueToUpdateBy / (sqrt(gradientSquared) + epsilon);
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
                double gradient = weight_gradients[layerIndex]->data[neuronIndex]->elements[weightIndex];
                // TODO rename valueToUpdateBy!!!!
                double valueToUpdateBy = -1 * (learningRate * gradient);
                double gradientSquared = gradient * gradient;

                // cache = rho * cache + (1 - rho) * gradient ** 2
                double fractionOfCache = rho *  currentLayer->weightCache->data[neuronIndex]->elements[weightIndex];
                double fractionOfGradientSquarred = (1 - rho) * gradientSquared;

                currentLayer->weightCache->data[neuronIndex]->elements[weightIndex] = fractionOfCache + fractionOfGradientSquarred;
                currentLayer->weights->data[neuronIndex]->elements[weightIndex] += valueToUpdateBy / (sqrt(fractionOfCache + fractionOfGradientSquarred) + epsilon);
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
                double gradient = weight_gradients[layerIndex]->data[neuronIndex]->elements[weightIndex];

                // m(t) = beta1 * m(t-1) + (1 – beta1) * g(t)
                double weightMomentum = beta1 * currentLayer->weightMomentums->data[neuronIndex]->elements[weightIndex] + (1 - beta1) * gradient;
                currentLayer->weightMomentums->data[neuronIndex]->elements[weightIndex] = weightMomentum;
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
                double weightCache = beta2 * currentLayer->weightCache->data[neuronIndex]->elements[weightIndex] + (1 - beta2) * pow(gradient, 2);
                currentLayer->weightCache->data[neuronIndex]->elements[weightIndex] = weightCache;

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
                double oldWeight = currentLayer->weights->data[neuronIndex]->elements[weightIndex];
                currentLayer->weights->data[neuronIndex]->elements[weightIndex] -= (learningRate * momentumCorrection) / (sqrt(cacheCorrection) + epsilon);
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
