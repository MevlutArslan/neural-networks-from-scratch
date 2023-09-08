#ifndef MODEL_H
#define MODEL_H

#include "../neural_network/nnetwork.h"
#include "../../libraries/gnuplot_i/gnuplot_i.h"
#include "../helper/data_processing.h"

typedef struct {
    // Arrays to store historical data for visualization using Gnuplot:
    double* loss_history;
    double* epoch_history;
    double* learning_rate_history;
    double* accuracy_history;
    int total_epochs;

    Matrix* training_data;
    Matrix* validation_data;
    Matrix* training_labels;
    Matrix* validation_labels;

    /*
        If you want to process sequentially use 0.
        If you want to process the entire dataset at once use 1.
        TODO: Implement ability to divide the dataset into multiple batches.
    */
    int num_batches;

    char* save_path;
} ModelData;


typedef struct Model{
    NNetwork* (*get_network)(struct Model* model);
    int (*preprocess_data)(ModelData* data);

    ModelData* data;
} Model;


// Register your models here
Model* create_wine_categorization_model();
NNetwork* wine_categorization_get_network(Model* model);
int wine_categorization_preprocess_data(ModelData* model_data);

Model* create_mnist_model();
NNetwork* mnist_get_network(Model* model);
int mnist_preprocess_data(ModelData* modelData);

// Linear Regression Model
Model* create_real_estate_model();

void train_model(Model* model, int should_save);
void validate_model(Model* model);

void free_model_data(ModelData* modelData);
void free_model(Model* model);
#endif
