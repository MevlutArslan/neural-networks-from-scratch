#include "../model.h"

Model* create_real_estate_model() {
    Model* model = (Model*)malloc(sizeof(Model));
    assert(model != NULL);

    model->get_network = real_estate_get_network;
    model->preprocess_data = real_estate_preprocess_data;

    model->data = (ModelData*) malloc(sizeof(ModelData));
    assert(model->data != NULL);

    model->data->total_epochs = 100;
    model->data->loss_history = calloc(model->data->total_epochs, sizeof(double));
    model->data->epoch_history = calloc(model->data->total_epochs, sizeof(double));
    model->data->learning_rate_history = calloc(model->data->total_epochs, sizeof(double));
    model->data->accuracy_history = calloc(model->data->total_epochs, sizeof(double));
    
    model->data->save_path = "real_estate_network";

    // TODO: Implement ability to divide the dataset into multiple batches.
    model->data->num_batches = 1; // I haven't abstracted out the sequential backward pass so I will use the batched one for now.

    return model;
}

void real_estate_preprocess_data(ModelData* model_data) {
    Data* real_estate_data = load_csv("/Users/mevlutarslan/Downloads/datasets/Real estate.csv");
    assert(real_estate_data != NULL);

    shuffle_rows(real_estate_data->data);

    float separation_factor = 0.7;

    int training_data_num_rows = real_estate_data->rows * separation_factor;
    // separate training and validation

    // last column is the y value (expected value) so I omit it out of the training values
    model_data->training_data = get_sub_matrix(real_estate_data->data, 0, training_data_num_rows, 1, real_estate_data->columns - 2);
    assert(model_data->training_data != NULL);
    // log_info("training data: %s", matrix_to_string(model_data->training_data));
    
    // get the last column and store it in a matrix of dimensions (training_data_num_rows x 1)
    model_data->training_labels = get_sub_matrix(real_estate_data->data, 0, training_data_num_rows, real_estate_data->columns - 1, real_estate_data->columns - 1);
    assert(model_data->training_labels != NULL);
    // log_info("training labels: %s", matrix_to_string(model_data->training_labels));
    
    model_data->validation_data = get_sub_matrix(real_estate_data->data, training_data_num_rows + 1, real_estate_data->rows - 1, 1, real_estate_data->columns -  2);
    assert(model_data->validation_data != NULL);
    // log_info("validation data: %s", matrix_to_string(model_data->validation_data));
    
    model_data->validation_labels = get_sub_matrix(real_estate_data->data, training_data_num_rows + 1, real_estate_data->rows - 1, real_estate_data->columns - 1, real_estate_data->columns -  1);
    assert(model_data->validation_labels != NULL);
    // log_info("validation labels: %s", matrix_to_string(model_data->validation_labels));

    for(int i = 0; i < model_data->training_data->columns; i++) {
        normalize_column(STANDARD_DEVIATION, model_data->training_data, i, 0);
    }

    for(int i = 0; i < model_data->validation_data->columns; i++) {
        normalize_column(STANDARD_DEVIATION, model_data->validation_data, i, 0);
    }
}

NNetwork* real_estate_get_network(Model* model) {
    real_estate_preprocess_data(model->data);

    return NULL;
}
