

#ifndef MNIST_H
#define MNIST_H

#include "../neural_network/nnetwork.h"
#include "../../libraries/gnuplot_i/gnuplot_i.h"
#include "../helper/data_processing.h"

NNetwork* get_network();
void train_network();
void validate_network();
int preprocess_data();
void plot_data(double* losses, double* storedSteps, double* learningRates, double* accuracies, double totalEpochs);
void plot_config();

#endif