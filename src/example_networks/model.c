#include "model.h"


void free_modelData(ModelData* model_data) {
    if(model_data) {
        free(model_data->loss_history);
        free(model_data->epoch_history);
        free(model_data->learning_rate_history);
        free(model_data->accuracy_history);

        free_matrix(model_data->training_data);
        free_matrix(model_data->validation_data);
        free_matrix(model_data->training_labels);
        free_matrix(model_data->validation_labels);
        
        // Add code here to free any other dynamically allocated members of ModelData

        free(model_data);
    }
}

void free_model(Model* model) {
    if(model) {
        free_modelData(model->data);
        free(model);
    }
}