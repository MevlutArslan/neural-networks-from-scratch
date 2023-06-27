#include "neuron.h"

Neuron* createNeuron(int numberOfInputs) {
    Neuron* n = malloc(sizeof (Neuron));

    // Generate random values between 0 and 1 for weight and bias
    n->weights = createVector(numberOfInputs);
    n->gradient = createVector(numberOfInputs + 1);
    for(int i = 0; i < numberOfInputs; i++) {
        n->weights->elements[i] = ((double)rand() / (double)RAND_MAX) * 2 - 1; ;
        n->gradient->elements[i] = 0.0f;
    }

    n->bias = 0;
    n->gradient->elements[numberOfInputs] = 0.0f;

    return n;
}

void applyGradientDescent(Neuron* neuron, double learningRate) {
    for(int i = 0; i < neuron->weights->size; i++) {
        neuron->weights->elements[i] -= learningRate * neuron->gradient->elements[i];
    }
}

void deleteNeuron(Neuron* n) {
    free(n);
}