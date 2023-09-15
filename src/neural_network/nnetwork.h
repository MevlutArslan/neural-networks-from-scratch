#ifndef NNETWORK_H
#define NNETWORK_H

#include "layer.h"
#include "../nmath/nmatrix.h"
#include "../nmath/nvector.h"
#include "../nmath/nmath.h"
#include "../helper/data_processing.h"
#include "loss_functions.h"
#include "../helper/constants.h"
#include "../../libraries/logger/log.h"
#include <stdio.h>

// struct ParallelForwardPassStruct{
    
// }
typedef struct {
    double learning_rate;
    
    int use_gradient_clipping;
    double gradient_clip_lower_bound;
    double gradient_clip_upper_bound;

    int use_learning_rate_decay;
    double learning_rate_decay_amount;

    int use_momentum;
    double momentum;

    enum OPTIMIZER optimizer;
    
    double epsilon;
    double rho;

    // ADAM
    double adam_beta1;
    double adam_beta2;

    // For l1 & l2 regularization
    int use_l1_regularization;
    int use_l2_regularization;

    Vector* l1_weight_lambdas; // <- can assign a different lambda to each layer 
    Vector* l1_bias_lambdas; // <- can assign a different lambda to each layer 

    Vector* l2_weight_lambdas;
    Vector* l2_bias_lambdas;
} OptimizationConfig;

typedef struct NNetwork{
    int num_layers;
    Layer** layers;

    LossFunctionType loss_fn;
    double loss;
    double accuracy;
    
    void (*optimization_algorithm)(struct NNetwork*, double, int);
    OptimizationConfig* optimization_config;
    int training_epoch;

    MatrixArray* weighted_sums;

    MatrixArray* batched_outputs; // Stores the output matrices for each layer. It's a 2D array where each row represents a layer, and each column represents the output of that layer.
    Matrix* output; // output to store sequential logic

    MatrixArray* weight_gradients;
    Vector** bias_gradients;
} NNetwork;

typedef struct {
    int num_layers;               // Number of layers in the network
    int* neurons_per_layer;        // Array of number of neurons per layer
    ActivationFunction* activation_fns;  // Array of activation functions for each layer
    double learning_rate;         // Learning rate for training the network
    LossFunctionType loss_fn;

    int num_features;
    int num_rows;

    OptimizationConfig* optimization_config;

    
} NetworkConfig;


NNetwork* create_network(const NetworkConfig* config);
void free_network(NNetwork* network);
void free_network_config(NetworkConfig* config);

void train_network(NNetwork* network, Matrix* training_data, Matrix* training_labels, int batch_size, int num_epochs, double learning_rate);

void forward_pass_sequential(NNetwork* network, Vector* input, Vector* output);
void forward_pass_batched(NNetwork* network, Matrix* input_matrix);

void calculate_loss(NNetwork* network, Matrix* yValues, Matrix* output);

void backpropagation_sequential(NNetwork* network, Vector* input, Vector* output, Vector* yValues);
void backpropagation_batched(NNetwork* network, Matrix* input_matrix, Matrix* y_values);

void calculate_weight_gradients(NNetwork* network, int layer_index, Matrix* loss_wrt_weightedsum, Matrix* wsum_wrt_weight);

void dump_network_config(NNetwork* network);

void sgd(NNetwork* network, double learningRate, int batch_size);
void adagrad(NNetwork* network, double learningRate, int batch_size);
void rms_prop(NNetwork* network, double learningRate, int batch_size);
void adam(NNetwork* network, double learningRate, int batch_size);

/*
    L1 regularization’s penalty is the sum of all the absolute values for the weights and biases.
    weights_penalty = lambda * sum(weights)
    bias_penalty = lambda * sum(biases)

    @param lambda dictates how much of an impact the regularization penalty carries.
*/
double calculate_l1_penalty(double lambda, const Vector* vector);
Vector* l1_derivative(double lambda, const Vector* vector);

/*
    L2 regularization’s penalty is the sum of the squared weights and biases.
    updated_weight = lambda * sum(weights^2)
    updated_bias = lambda * sum(biases ^ 2)

    @param lambda variable dictates how much of an impact the regularization penalty carries.
*/
double calculate_l2_penalty(double lambda, const Vector* vector);
Vector* l2_derivative(double lambda, const Vector* vector);

double accuracy(Matrix* targets, Matrix* outputs);

char* serialize_network(const NNetwork* network);
NNetwork* deserialize_network(cJSON* json);

int save_network(char* path, NNetwork* network);
NNetwork* load_network(char* path);
#endif