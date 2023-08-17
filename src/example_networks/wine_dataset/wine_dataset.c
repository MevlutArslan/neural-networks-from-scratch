#include "wine_dataset.h"

NNetwork* wine_categorization_get_network(Model* model);
void wine_categorization_train_network(Model* model);
void wine_categorization_validate_network(Model* model);
int wine_categorization_preprocess_data(ModelData* modelData);
void wine_categorization_plot_data(ModelData* modelData);
void wine_categorization_plot_config();

// doesnt seem to contain any leaks so far
Model* create_wine_categorization_model() {
    Model* model = malloc(sizeof(Model));

    model->get_network = wine_categorization_get_network;
    model->train_network = wine_categorization_train_network;
    model->validate_network = wine_categorization_validate_network;
    model->preprocess_data = wine_categorization_preprocess_data;
    model->plot_data = wine_categorization_plot_data;
    model->plot_config = wine_categorization_plot_config;

    int totalEpochs = 250;
    model->data = (ModelData*) malloc(sizeof(ModelData));
    model->data->totalEpochs = totalEpochs;
    model->data->losses = calloc(totalEpochs, sizeof(double));
    model->data->epochs = calloc(totalEpochs, sizeof(double));
    model->data->learningRates = calloc(totalEpochs, sizeof(double));
    model->data->accuracies = calloc(totalEpochs, sizeof(double));
    model->data->path = "wine_dataset_network";

    // model->thread_pool = create_thread_pool(6);

    return model;
}

// doesnt seem to contain any leaks.
OptimizationConfig wine_categorization_create_optimizer(int optimizer) {
    OptimizationConfig optimizationConfig;
    optimizationConfig.optimizer = optimizer;

    // Learning Rate Decay
    optimizationConfig.shouldUseLearningRateDecay = 1;
    optimizationConfig.shouldUseGradientClipping = 0;
    // optimizationConfig.rho = 0.9;
    optimizationConfig.epsilon = 1e-8;
    optimizationConfig.beta1 = 0.9;
    optimizationConfig.beta2 = 0.999;


    return optimizationConfig;
}


int wine_categorization_preprocess_data(ModelData* modelData) {
    Data* wineData = load_csv("/home/mvlcfr/datasets/wine_dataset/wine_with_headers.csv");
    
    if(wineData == NULL) {
        log_error("%s", "Failed to load Wine Data");
        return -1;
    }

    shuffle_rows(wineData->data);
    // log_info("shuffled rows: %s", matrix_to_string(wineData->data));
    // the amount we want to divide the dataset between training and validation data
    double divisionPercentage = 0.8;

    int targetColumn = 0;
    
    int trainingDataSize = wineData->rows * divisionPercentage;

    // TODO: OPTIMIZE
    Matrix* trainingData = get_sub_matrix(wineData->data, 0, trainingDataSize - 1, 0, wineData->data->columns - 1);
    modelData->trainingData = get_sub_matrix_except_column(trainingData, 0, trainingData->rows - 1, 0, trainingData->columns - 1, targetColumn);
    
    Vector* yValues_training = extractYValues(trainingData, targetColumn);
    modelData->yValues_Training = oneHotEncode(yValues_training, 3);

    
    for(int colIndex = 0; colIndex < modelData->trainingData->columns; colIndex++) {
        normalizeColumn_standard_deviation(modelData->trainingData, colIndex);
    }
    
    Matrix* validationData = get_sub_matrix(wineData->data, modelData->trainingData->rows, wineData->rows - 1, 0, wineData->data->columns - 1);
    modelData->validationData = get_sub_matrix_except_column(validationData, 0, validationData->rows -1, 0, validationData->columns - 1, targetColumn);

    Vector* yValues_validation = extractYValues(validationData, targetColumn);
    modelData->yValues_Testing = oneHotEncode(yValues_validation, 3);
    
    for(int colIndex = 0; colIndex < modelData->validationData->columns; colIndex++) {
        normalizeColumn_standard_deviation(modelData->validationData, colIndex);
    }

    free_matrix(trainingData);
    free_matrix(validationData);
    free_vector(yValues_training);
    free_vector(yValues_validation);
    free_data(wineData);

    return 1;
}


NNetwork* wine_categorization_get_network(Model* model) {
    if(model->preprocess_data(model->data) != 1) {
        log_error("%s", "Failed to complete preprocessing of Wine Categorization data!");
    }
    NetworkConfig config;
    config.numLayers = 2;
    config.neuronsPerLayer = malloc(sizeof(int) * config.numLayers);
    config.neuronsPerLayer[0] = 2;
    config.neuronsPerLayer[1] = 3;

    config.num_rows = model->data->trainingData->rows;
    config.num_features = model->data->trainingData->columns;

    OptimizationConfig optimizationConfig = wine_categorization_create_optimizer(ADAM);

    // if you want to use l1 and/or l2 regularization you need to set the size to config.numLayers and 
    // fill these vectors with the lambda values you want
    config.weightLambdas = create_vector(0);
    config.biasLambdas = create_vector(0);

    if(config.weightLambdas->size > 0 ){
        fill_vector(config.weightLambdas, 1e-5);
    }

    if(config.biasLambdas->size > 0 ){
        fill_vector(config.biasLambdas, 1e-3);
    }

    config.activationFunctions = calloc(config.numLayers, sizeof(ActivationFunction));
    
    config.optimizationConfig = malloc(sizeof(OptimizationConfig));
    memcpy(config.optimizationConfig, &optimizationConfig, sizeof(OptimizationConfig));
    
    for (int i = 0; i < config.numLayers - 1; i++) {
        memcpy(&config.activationFunctions[i], &LEAKY_RELU, sizeof(ActivationFunction));
    }

    // output layer's activation
    memcpy(&config.activationFunctions[config.numLayers - 1], &SOFTMAX, sizeof(ActivationFunction));

    config.lossFunction = malloc(sizeof(LossFunction));
    memcpy(&config.lossFunction->loss_function, &CATEGORICAL_CROSS_ENTROPY, sizeof(LossFunction));

    NNetwork* network = create_network(&config);

    free_network_config(&config);
    model->plot_config();

    return network;
}

void wine_categorization_train_network(Model* model) {
    NNetwork* network = wine_categorization_get_network(model);

    if(network == NULL) {
        log_error("%s", "Error creating network!");
        return;
    }

    ModelData* modelData = model->data;
    
    // default rate of keras -> 0.001
    // kaparthy's recommendation for adam: 0.0003
    double learningRate = 0.01;
    double currentLearningRate = learningRate;
    int step = 1;

    network->optimizationConfig->learningRateDecayAmount = learningRate / modelData->totalEpochs;

    double minLoss = __DBL_MAX__;
    double maxAccuracy = 0.0;

    log_debug("Starting training with learning rate of: %f for %d epochs.", learningRate,  modelData->totalEpochs);
    while(step < 2) {
        modelData->learningRates[step] = currentLearningRate;

        forward_pass_batched(network, modelData->trainingData); 
        backpropagation_batched(network, modelData->trainingData, modelData->yValues_Training);

        calculate_loss(network, modelData->yValues_Training);

        // if(network->optimizationConfig->shouldUseLearningRateDecay == 1) {
        //     double decayRate = network->optimizationConfig->learningRateDecayAmount;
        //     currentLearningRate = currentLearningRate * (1 / (1.0 + (decayRate * (double)step)));
        // }

        // network->currentStep = step;
        // network->optimizer(network, currentLearningRate);

        // if(step == 1 || step % 10 == 0){
        //     log_debug("Step: %d, Accuracy: %f, Loss: %f \n", step, network->accuracy, network->loss);  
        // }
        minLoss = fmin(minLoss, network->loss);
        
        maxAccuracy = fmax(maxAccuracy, network->accuracy);

        // // modelData->losses[step] = network->loss;
        // // modelData->epochs[step] = step;
        // // modelData->accuracies[step] = network->accuracy;
        step++;
        // // // Clear the gradients
        // for(int layerIndex = 0; layerIndex < network->layerCount; layerIndex++) {
        //     fill_matrix(network->weight_gradients[layerIndex], 0.0f);
        //     fill_vector(network->bias_gradients[layerIndex], 0.0f);
        // }
    }

    log_info("Minimum loss during training: %f \n", minLoss);
    log_info("Maximum accuracy during training: %f \n", maxAccuracy);
    
    // save_network(modelData->path, network);

    // model->plot_data(model->data);
    free_network(network);
}

void wine_categorization_validate_network(Model* model) {
    NNetwork* network = load_network(model->data->path);

    forward_pass_batched(network, model->data->validationData);
    calculate_loss(network, model->data->yValues_Testing);

    log_info("Validation Loss: %f", network->loss);
    log_info("Validation Accuracy: %f", network->accuracy);

    free_network(network);
}

gnuplot_ctrl* loss_step_plot;
gnuplot_ctrl* accuracy_step_plot;
gnuplot_ctrl* learning_rate_step_plot;
void wine_categorization_plot_data(ModelData* modelData) {
    gnuplot_plot_xy(loss_step_plot, modelData->epochs, modelData->losses, modelData->totalEpochs, "loss/step");
    gnuplot_plot_xy(accuracy_step_plot, modelData->epochs, modelData->accuracies, modelData->totalEpochs, "accuracy/step");
    gnuplot_plot_xy(learning_rate_step_plot, modelData->epochs, modelData->learningRates, modelData->totalEpochs, "learning rate/step");

    printf("press Enter to close the plots!");
    getchar();

    gnuplot_close(loss_step_plot);
    gnuplot_close(accuracy_step_plot);
    gnuplot_close(learning_rate_step_plot);
}

void wine_categorization_plot_config() {
    loss_step_plot = gnuplot_init();

    gnuplot_set_xlabel(loss_step_plot, "step");
    gnuplot_set_ylabel(loss_step_plot, "loss");
    
    accuracy_step_plot = gnuplot_init();
    
    gnuplot_set_xlabel(accuracy_step_plot, "step");
    gnuplot_set_ylabel(accuracy_step_plot, "accuracy");

    learning_rate_step_plot = gnuplot_init();
    
    gnuplot_set_xlabel(learning_rate_step_plot, "step");
    gnuplot_set_ylabel(learning_rate_step_plot, "learning rate");

}

