#ifndef MODEL_H
#define MODEL_H

#include "../neural_network/nnetwork.h"
#include "../../libraries/gnuplot_i/gnuplot_i.h"
#include "../helper/data_processing.h"

typedef struct {
    double* losses;
    double* epochs;
    double* learningRates;
    double* accuracies;
    int totalEpochs;

    Matrix* trainingData;
    Matrix* validationData;
    Matrix* yValues_Training;
    Matrix* yValues_Testing;

    char* path;
} ModelData;


typedef struct {
    NNetwork* (*get_network)(struct Model* model);
    void (*train_network)(struct Model* data);
    void (*validate_network)(struct  Model* data);
    int (*preprocess_data)(ModelData* data);
    void (*plot_data)(ModelData* data);
    void (*plot_config)();
    ModelData* data;
} Model;

void free_modelData(ModelData* modelData);
void free_model(Model* model);
#endif

