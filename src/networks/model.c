#include "model.h"

void free_modelData(ModelData* modelData) {
    if(modelData) {
        free(modelData->loss_history);
        free(modelData->epoch_history);
        free(modelData->learning_rate_history);
        free(modelData->accuracy_history);

        free_matrix(modelData->training_data);
        free_matrix(modelData->validation_data);
        free_matrix(modelData->training_labels);
        free_matrix(modelData->validation_labels);
        
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