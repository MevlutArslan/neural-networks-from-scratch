#include "model.h"

void train_model(Model* model, int should_save) {
    NNetwork* network = model->get_network(model);

    if(network == NULL) {
        log_error("%s", "Error creating network!");
        return;
    }
    ModelData* model_data = model->data;

    log_info("Starting to train model!");

    train_network(network, model_data->training_data, model_data->training_labels, model_data->num_batches, model->data->total_epochs, network->optimization_config->learning_rate);

    if(should_save == TRUE) {
        save_network(model_data->save_path, network);
    }
    free_network(network);
}
/*
    Using batched or sequential processing does not matter in terms of the result, (because the weights & biases are from the same source)
    for the sake of simplicity I chose to implement validation using batched processing.
*/
void validate_model(Model* model) {
    NNetwork* network = load_network(model->data->save_path);

    forward_pass_batched(network, model->data->validation_data);
    calculate_loss(network, model->data->validation_labels, network->batched_outputs[network->num_layers - 1]);

    log_info("Validation Loss: %f", network->loss);
    log_info("Validation Accuracy: %f", network->accuracy);

    free_network(network);
}

void free_model_data(ModelData* modelData) {
    if(modelData == NULL) {
        return;
    }

    free(modelData->loss_history);
    free(modelData->epoch_history);
    free(modelData->learning_rate_history);
    free(modelData->accuracy_history);

    free_matrix(modelData->training_data);
    free_matrix(modelData->validation_data);
    free_matrix(modelData->training_labels);
    free_matrix(modelData->validation_labels);
        
    free(modelData);
}

void free_model(Model* model) {
    if(model == NULL) {
        return;
    }
    free_model_data(model->data);
    free(model);
}
