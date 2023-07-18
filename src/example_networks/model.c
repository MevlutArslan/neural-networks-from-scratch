#include "model.h"


void free_modelData(ModelData* modelData) {
    if(modelData) {
        free(modelData->losses);
        free(modelData->epochs);
        free(modelData->learningRates);
        free(modelData->accuracies);

        free_matrix(modelData->trainingData);
        free_matrix(modelData->validationData);
        free_matrix(modelData->yValues_Training);
        free_matrix(modelData->yValues_Testing);
        
        // Add code here to free any other dynamically allocated members of ModelData

        free(modelData);
    }
}

void free_model(Model* model) {
    if(model) {
        free_modelData(model->data);
        free(model);
    }
}