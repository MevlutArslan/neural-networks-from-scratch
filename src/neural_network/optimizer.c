#include "nnetwork.h"

void sgd(NNetwork* network, double learningRate) {
    double momentum = network->optimizationConfig->momentum;

    if(network->optimizationConfig->shouldUseMomentum == 1 && momentum == 0) {
        printf("MOMENTUM HASN'T BEEN SET! \n");
        return;
    }

    for(int layerIndex = 0; layerIndex < network->layerCount; layerIndex++) {
        Layer* currentLayer = network->layers[layerIndex];

        for(int neuronIndex = 0; neuronIndex < currentLayer->neuronCount; neuronIndex++) {
            double biasGradient = currentLayer->biasGradients->elements[neuronIndex];
            double biasValueToUpdateBy = -1 * (learningRate * biasGradient);
            for(int weightIndex = 0; weightIndex < currentLayer->weights->columns; weightIndex++) {
                double gradient = currentLayer->gradients->data[neuronIndex]->elements[weightIndex];
                double valueToUpdateBy = -1 * (learningRate * gradient);
                    
                if(network->optimizationConfig->shouldUseMomentum == 1) {
                    double momentumUpdate = momentum * currentLayer->weightMomentums->data[neuronIndex]->elements[weightIndex];
                    valueToUpdateBy = momentumUpdate - (learningRate * gradient);

                    currentLayer->weightMomentums->data[neuronIndex]->elements[weightIndex] = valueToUpdateBy;
                }

                currentLayer->weights->data[neuronIndex]->elements[weightIndex] += valueToUpdateBy;
            }
            if(network->optimizationConfig->shouldUseMomentum == 1) {
                double momentumUpdate = momentum * currentLayer->biasMomentums->elements[neuronIndex];
                biasValueToUpdateBy = momentumUpdate - (learningRate * biasGradient);
                currentLayer->biasMomentums->elements[neuronIndex] = biasValueToUpdateBy;
            }

            currentLayer->biases->elements[neuronIndex] += biasValueToUpdateBy;
        }
    }
}

void adagrad(NNetwork* network, double learningRate) {
    double epsilon = network->optimizationConfig->epsilon;

    if(epsilon == 0) {
        printf("EPSILON HAS NOT BEEN SET! \n");
        return;
    }
    
    for(int layerIndex = 0; layerIndex < network->layerCount; layerIndex++) {
        Layer* currentLayer = network->layers[layerIndex];

        for(int neuronIndex = 0; neuronIndex < currentLayer->neuronCount; neuronIndex++) {
            double biasGradient = currentLayer->biasGradients->elements[neuronIndex];
            // TODO rename biasValueToUpdateBy!!!!
            double biasValueToUpdateBy = -1 * (learningRate * biasGradient);
            double biasGradientSquared = biasGradient * biasGradient;
            
            for(int weightIndex = 0; weightIndex < currentLayer->weights->columns; weightIndex++) {
                double gradient = currentLayer->gradients->data[neuronIndex]->elements[weightIndex];
                // TODO rename valueToUpdateBy!!!!
                double valueToUpdateBy = -1 * (learningRate * gradient);
                double gradientSquared = gradient * gradient;

                currentLayer->weightCache->data[neuronIndex]->elements[weightIndex] = gradientSquared;
                currentLayer->weights->data[neuronIndex]->elements[weightIndex] += valueToUpdateBy / (sqrt(gradientSquared) + epsilon);
            }

            currentLayer->biasCache->elements[neuronIndex] = biasGradientSquared;
            currentLayer->biases->elements[neuronIndex] += biasValueToUpdateBy / (sqrt(biasGradientSquared) + epsilon);
        }
    }
}

void rms_prop(NNetwork* network, double learningRate) {
    double rho = network->optimizationConfig->rho;
    double epsilon = network->optimizationConfig->epsilon;

    if(rho == 0) {
        printf("RHO HAS NOT BEEN SET! \n");
        return;
    }

    if(epsilon == 0) {
        printf("EPSILON HAS NOT BEEN SET! \n");
        return;
    }
    

    for(int layerIndex = 0; layerIndex < network->layerCount; layerIndex++) {
        Layer* currentLayer = network->layers[layerIndex];

        for(int neuronIndex = 0; neuronIndex < currentLayer->neuronCount; neuronIndex++) {
            double biasGradient = currentLayer->biasGradients->elements[neuronIndex];
            // TODO rename biasValueToUpdateBy!!!!
            double biasValueToUpdateBy = -1 * (learningRate * biasGradient);
            double biasGradientSquared = biasGradient * biasGradient;

            for(int weightIndex = 0; weightIndex < currentLayer->weights->columns; weightIndex++) {
                double gradient = currentLayer->gradients->data[neuronIndex]->elements[weightIndex];
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

    if(beta1 == 0) {
        printf("BETA_1 HAS NOT BEEN SET! \n");
        return;
    }

    if(beta2 == 0) {
        printf("BETA_2 HAS NOT BEEN SET! \n");
        return;
    }

    if(epsilon == 0) {
        printf("EPSILON HAS NOT BEEN SET! \n");
        return;
    }

    for(int layerIndex = 0; layerIndex < network->layerCount; layerIndex++) {
        Layer* currentLayer = network->layers[layerIndex];

        for(int neuronIndex = 0; neuronIndex < currentLayer->neuronCount; neuronIndex++) {
            for(int weightIndex = 0; weightIndex < currentLayer->weights->columns; weightIndex++) {
                double gradient = currentLayer->gradients->data[neuronIndex]->elements[weightIndex];
                
                double weightMomentum = beta1 * currentLayer->weightMomentums->data[neuronIndex]->elements[weightIndex];
                weightMomentum += (1 - beta1) * gradient;

                currentLayer->weightMomentums->data[neuronIndex]->elements[weightIndex] = weightMomentum;

                double momentumCorrection = weightMomentum / (1 - pow(beta1, network->currentStep + 1));
                currentLayer->weightMomentums->data[neuronIndex]->elements[weightIndex] = momentumCorrection;

                double weightCache = beta2 * currentLayer->weightCache->data[neuronIndex]->elements[weightIndex];
                weightCache += ((1 - beta2) * pow(gradient, 2));

                double cacheCorrection = weightCache;
                cacheCorrection /= (1 - pow(beta2, network->currentStep+1));
                currentLayer->weightCache->data[neuronIndex]->elements[weightIndex] = cacheCorrection;

                double weightModificiation = (learningRate * momentumCorrection);
                weightModificiation /= sqrt(cacheCorrection) + epsilon;
                weightModificiation *= -1;
                
                currentLayer->weights->data[neuronIndex]->elements[weightIndex] += weightModificiation;
            }
            double biasGradient = currentLayer->biasGradients->elements[neuronIndex];

            // BIAS UPDATE

            // Momentum calculations
            double biasMomentum = beta1 * currentLayer->biasMomentums->elements[neuronIndex];
            biasMomentum += (1 - beta1) * biasGradient;
            currentLayer->biasMomentums->elements[neuronIndex] = biasMomentum;

            double momentumCorrection = biasMomentum / (1 - pow(beta1, network->currentStep + 1));

            // Cache calculations
            double biasCache = beta2 * currentLayer->biasCache->elements[neuronIndex];
            biasCache += (1 - beta2) * pow(biasGradient, 2);
            currentLayer->biasCache->elements[neuronIndex] = biasCache;

            double cacheCorrection = biasCache / (1 - pow(beta2, network->currentStep + 1));

            // Weight modification
            double biasModification = (learningRate * momentumCorrection);
            biasModification /= sqrt(cacheCorrection) + epsilon;
            biasModification *= -1;

            currentLayer->biases->elements[neuronIndex] += biasModification;
        }
    }
    
}