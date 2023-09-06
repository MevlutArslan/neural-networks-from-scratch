#ifndef MODEL_H
#define MODEL_H

#include "../neural_network/nnetwork.h"
#include "../../libraries/gnuplot_i/gnuplot_i.h"
#include "../helper/data_processing.h"

typedef struct {
    double* loss_history;
    double* epoch_history;
    double* learning_rate_history;
    double* accuracy_history;
    int total_epochs;

    Matrix* training_data;
    Matrix* validation_data;
    Matrix* training_labels;
    Matrix* validation_labels;

    char* path;
} ModelData;


typedef struct Model{
    NNetwork* (*get_network)(struct Model* model);
    void (*train_network)(struct Model* data);
    void (*validate_network)(struct  Model* data);
    int (*preprocess_data)(ModelData* data);
    void (*plot_data)(ModelData* data);
    void (*plot_config)();
    ModelData* data;
} Model;


// Register your models here
Model* create_wine_categorization_model();
Model* create_mnist_model();


void free_modelData(ModelData* modelData);
void free_model(Model* model);
#endif
