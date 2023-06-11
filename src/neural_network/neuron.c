#include "neuron.h"

Neuron* createNeuron(int numberOfInputs) {
    Neuron* n = malloc(sizeof (Neuron));

    // Generate random values between 0 and 1 for weight and bias
    n->weights = createVector(numberOfInputs);

    for(int i = 0; i < numberOfInputs; i++) {
        n->weights->elements[i] = (double)rand() / (double)RAND_MAX ;
    }
    n->bias = 0;

    return n;
}


void deleteNeuron(Neuron* n) {
    free(n);
}