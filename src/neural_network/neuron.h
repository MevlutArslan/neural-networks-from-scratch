#ifndef NEURON_H
#define NEURON_H

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include "../nmath/nvector.h"

typedef struct {
    Vector* weights;
    double bias;
} Neuron;


Neuron* createNeuron(int numberOfInputs);
void deleteNeuron(Neuron* neuron);

#endif