#ifndef NEURON_H
#define NEURON_H

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct {
    double weight;
    double bias;
} Neuron;


Neuron* createNeuron();
void deleteNeuron();

#endif