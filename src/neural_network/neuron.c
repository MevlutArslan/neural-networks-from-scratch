#include "neuron.h"

Neuron* createNeuron() {
    Neuron* n = malloc(sizeof (Neuron));

    // Generate random values between 0 and 1 for weight and bias
    n->weight = (double)rand() / (double)RAND_MAX ;
    n->bias = 0;

    return n;
}

void deleteNeuron(Neuron* n) {
    free(n);
}