#include "nnetwork.h"

void sgd(NNetwork* network, double learningRate) {
    double momentum = network->optimizationConfig->momentum;

    if(network->optimizationConfig->shouldUseMomentum == 1 && momentum == 0) {
        printf("MOMENTUM HASN'T BEEN SET!");
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
        printf("EPSILON HAS NOT BEEN SET!");
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
        printf("RHO HAS NOT BEEN SET!");
        return;
    }

    if(epsilon == 0) {
        printf("EPSILON HAS NOT BEEN SET!");
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