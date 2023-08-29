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
#include "../helper/thread_pool.h"

// struct ParallelForwardPassStruct{
    
// }
typedef struct {
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
} OptimizationConfig;

typedef struct NNetwork{
    int num_layers;
    Layer** layers;

    LossFunctionType loss_fn;
    double loss;
    double accuracy;
    
    void (*optimization_algorithm)(struct NNetwork*, double);
    OptimizationConfig* optimization_config;
    int training_epoch;

    Matrix** weighted_sums;

    Matrix** layer_outputs; // Stores the output matrices for each layer. It's a 2D array where each row represents a layer, and each column represents the output of that layer.

    Matrix** weight_gradients;
    Vector** bias_gradients;

    struct ThreadPool* thread_pool;
} NNetwork;

typedef struct {
    int numLayers;               // Number of layers in the network
    int* neurons_per_layer;        // Array of number of neurons per layer
    ActivationFunction* activation_fns;  // Array of activation functions for each layer
    double learning_rate;         // Learning rate for training the network
    LossFunctionType loss_fn;

    int num_features;
    int num_rows;

    OptimizationConfig* optimization_config;

    // For l1 & l2 regularization
    Vector* weight_lambdas;
    Vector* bias_lambdas;
} NetworkConfig;

NNetwork* create_network(const NetworkConfig* config);
void init_network_memory(NNetwork* network, int num_rows);

void free_network(NNetwork* network);
void free_network_config(NetworkConfig* config);

void forward_pass_batched(NNetwork* network, Matrix* input_matrix);

void calculate_loss(NNetwork* network, Matrix* yValues);

void backpropagation_batched(NNetwork* network, Matrix* input_matrix, Matrix* y_values);
void calculate_weight_gradients(NNetwork* network, int layer_index, Matrix* loss_wrt_weightedsum, Matrix* wsum_wrt_weight);

#ifdef __cplusplus
extern "C" {
#endif

void calculate_weight_gradients_cuda(NNetwork* network, int layer_index, Matrix* loss_wrt_weightedsum, Matrix* wsum_wrt_weight); 

#ifdef __cplusplus
}
#endif

void dump_network_config(NNetwork* network);

void sgd(NNetwork* network, double learningRate);
void adagrad(NNetwork* network, double learningRate);
void rms_prop(NNetwork* network, double learningRate);
void adam(NNetwork* network, double learningRate);

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