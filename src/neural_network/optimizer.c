#include "nnetwork.h"

void sgd(NNetwork* network, double learningRate) {
    for(int layerIndex = 0; layerIndex < network->layerCount; layerIndex++) {
        Layer* currentLayer = network->layers[layerIndex];

        for(int neuronIndex = 0; neuronIndex < currentLayer->neuronCount; neuronIndex++) {
            double biasGradient = currentLayer->biasGradients->elements[neuronIndex];
            double biasValueToUpdateBy = -1 * (learningRate * biasGradient);
            for(int weightIndex = 0; weightIndex < currentLayer->weights->columns; weightIndex++) {
                double gradient = currentLayer->gradients->data[neuronIndex]->elements[weightIndex];
                double valueToUpdateBy = -1 * (learningRate * gradient);
                    
                if(network->optimizationConfig->shouldUseMomentum == 1) {
                    double momentumUpdate = network->optimizationConfig->momentum * currentLayer->weightMomentums->data[neuronIndex]->elements[weightIndex];
                    valueToUpdateBy = momentumUpdate - (learningRate * gradient);

                    currentLayer->weightMomentums->data[neuronIndex]->elements[weightIndex] = valueToUpdateBy;
                }

                currentLayer->weights->data[neuronIndex]->elements[weightIndex] += valueToUpdateBy;
            }
            if(network->optimizationConfig->shouldUseMomentum == 1) {
                double momentumUpdate = network->optimizationConfig->momentum * currentLayer->biasMomentums->elements[neuronIndex];
                biasValueToUpdateBy = momentumUpdate - (learningRate * biasGradient);
                currentLayer->biasMomentums->elements[neuronIndex] = biasValueToUpdateBy;
            }

            currentLayer->biases->elements[neuronIndex] += biasValueToUpdateBy;
        }
    }
}

void adagrad(NNetwork* network, double learningRate) {
    for(int layerIndex = 0; layerIndex < network->layerCount; layerIndex++) {
        Layer* currentLayer = network->layers[layerIndex];

        for(int neuronIndex = 0; neuronIndex < currentLayer->neuronCount; neuronIndex++) {
            double biasGradient = currentLayer->biasGradients->elements[neuronIndex];
            double biasValueToUpdateBy = -1 * (learningRate * biasGradient);
            
            for(int weightIndex = 0; weightIndex < currentLayer->weights->columns; weightIndex++) {
                double gradient = currentLayer->gradients->data[neuronIndex]->elements[weightIndex];
                double valueToUpdateBy = -1 * (learningRate * gradient);
                double gradientSquared = gradient * gradient;

                currentLayer->weightCache_Adagrad->data[neuronIndex]->elements[weightIndex] = gradientSquared;
                currentLayer->weights->data[neuronIndex]->elements[weightIndex] += valueToUpdateBy / (sqrt(gradientSquared) + network->optimizationConfig->epsilon);
            }

            currentLayer->biases->elements[neuronIndex] = biasGradient * biasGradient;
            currentLayer->biases->elements[neuronIndex] += biasValueToUpdateBy;
        }
    }
}
