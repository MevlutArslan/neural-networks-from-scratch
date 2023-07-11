#include "nnetwork.h"

/*
    createNetwork works by getting the config and the inputs.
    It allocates memory for the Neural Network based on the struct 

    typedef struct {
        Layer* start;
        Layer* end;
    } NNetwork;

    Where each layer holds information like :
    * number of neurons
    * pointer to a vector of inputs
    * Pointer to an array of Neurons
    * Pointer to a vector of outputs
    * Pointer to an activation function
    * Pointer to the next layer (Layers can classified as linked lists)
    * Pointer to the prev layer (Layers can classified as linked lists)

    
    we loop config.number of layers times and create each layer.
    we mark the first layer to be network's start layer and the last layer to be the end layer
*/
NNetwork* createNetwork(const NetworkConfig* config) {
    NNetwork* network = malloc(sizeof(NNetwork));
    network->data = config->data;
    network->layerCount = config->numLayers;
    network->layers = malloc(network->layerCount * sizeof(Layer));

   for (int i = 0; i < config->numLayers; i++) {
        LayerConfig layerConfig;
        layerConfig.inputSize = i == 0 ? config->data->trainingData->columns: network->layers[i - 1]->output->size;
        layerConfig.neuronCount = config->neuronsPerLayer[i];
        layerConfig.activationFunction = &config->activationFunctions[i];
        layerConfig.willUseMomentum = config->optimizationConfig->shouldUseMomentum;
        layerConfig.optimizer = config->optimizationConfig->optimizer;
        
        Layer* layer = createLayer(&layerConfig);
        network->layers[i] = layer;
    }

    network->lossFunction = config->lossFunction;
    network->optimizationConfig = config->optimizationConfig;

    switch (network->optimizationConfig->optimizer) {
        case SGD:
            network->optimizer = sgd;
            break;
        case ADAGRAD:
            network->optimizer = adagrad;
            break;
        case RMS_PROP:
            network->optimizer = rms_prop;
            break;
        case ADAM:
            network->optimizer = adam;
            break;
        default:
            break;
    }
    network->data->trainingOutputs = createMatrix(network->data->trainingData->rows, network->layers[network->layerCount - 1]->neuronCount);

    log_info("Created Network:");
    dumpNetworkState(network);

    return network;
}

void forwardPass(NNetwork* network, Matrix* input, Matrix* output) {
    if((input->rows != output->rows)) {
        printf("INPUT AND OUTPUT MATRIX SIZE MISMATCH! \n");
        return;
    }
    for (int i = 0; i < 5; i++) {
        network->layers[0]->input = input->data[i];
        for (int layerIndex = 0; layerIndex < 1; layerIndex++) {
            Layer* currentLayer = network->layers[layerIndex];

            Vector* weightedSum = dot_product(currentLayer->weights, currentLayer->input);
    
            currentLayer->output = vector_addition(weightedSum, currentLayer->biases);;
            currentLayer->weightedSums = copyVector(currentLayer->output);
            
            currentLayer->activationFunction->activation(currentLayer->output);
            
            if(layerIndex != network->layerCount - 1) {
                network->layers[layerIndex + 1]->input = currentLayer->output;
            }
        }
        output->data[i] = copyVector(network->layers[network->layerCount - 1]->output);
    }
}

void backpropagation(NNetwork* network) {
    network->loss = network->lossFunction->loss_function(network->data->yValues, network->data->trainingOutputs);
    
    // Clear the gradients
    for(int layerIndex = 0; layerIndex < network->layerCount; layerIndex++) {
        fillMatrix(network->layers[layerIndex]->gradients, 0);
    }

    Layer* outputLayer = network->layers[network->layerCount - 1];

    // for each output
    for(int outputIndex = 0; outputIndex < 1; outputIndex++) {
        
        // the output layer's step
        int layerIndex = network->layerCount - 1;
        Layer* currentLayer = network->layers[layerIndex];
        for(int outputNeuronIndex = 0; outputNeuronIndex < outputLayer->neuronCount; outputNeuronIndex++) {
            double prediction = network->data->trainingOutputs->data[outputIndex]->elements[outputNeuronIndex];

            double target = network->data->yValues->data[outputIndex]->elements[outputNeuronIndex];

            double dLoss_dOutput = network->lossFunction->derivative(target, prediction);

            double dOutput_dWeightedSum = 0.0f;
            double dLoss_dWeightedSum = 0.0f;

            // if(currentLayer->activationFunction->activation == &softmax && network->lossFunction == &crossEntropyLoss) {
                dLoss_dWeightedSum = prediction - target;
            // }else {
            //     printf("SHOULDNT BE HERE!");
            //     dOutput_dWeightedSum = currentLayer->activationFunction->derivative(currentLayer->weightedSums->elements[outputNeuronIndex]);
            //     dLoss_dWeightedSum = dLoss_dOutput * dOutput_dWeightedSum;
            // }

            for(int weightIndex = 0; weightIndex < currentLayer->weights->columns; weightIndex++) {
                double dWeightedSum_dWeight = 0.0f;
                if(layerIndex == 0) {
                    dWeightedSum_dWeight= network->layers[layerIndex]->input->elements[weightIndex];
                }else {
                    dWeightedSum_dWeight = network->layers[layerIndex-1]->output->elements[weightIndex];
                }

                double dLoss_dWeight = dLoss_dWeightedSum * dWeightedSum_dWeight;
                
                currentLayer->gradients->data[outputNeuronIndex]->elements[weightIndex] += dLoss_dWeight;

                if(network->optimizationConfig->shouldUseGradientClipping == 1) {
                    if(currentLayer->gradients->data[outputNeuronIndex]->elements[weightIndex] < network->optimizationConfig->gradientClippingLowerBound) {
                        currentLayer->gradients->data[outputNeuronIndex]->elements[weightIndex] = network->optimizationConfig->gradientClippingLowerBound;
                    }else if(currentLayer->gradients->data[outputNeuronIndex]->elements[weightIndex] > network->optimizationConfig->gradientClippingUpperBound) {
                        currentLayer->gradients->data[outputNeuronIndex]->elements[weightIndex] = network->optimizationConfig->gradientClippingUpperBound;
                    }
                }              
            }
            
            if(network->optimizationConfig->shouldUseGradientClipping == 1) {
                if(dLoss_dWeightedSum < network->optimizationConfig->gradientClippingLowerBound) {
                    currentLayer->biasGradients->elements[outputNeuronIndex] = network->optimizationConfig->gradientClippingLowerBound;
                }else if(dLoss_dWeightedSum > network->optimizationConfig->gradientClippingUpperBound) {
                    currentLayer->biasGradients->elements[outputNeuronIndex] = network->optimizationConfig->gradientClippingUpperBound;                        
                }
            }else {
                currentLayer->biasGradients->elements[outputNeuronIndex] = dLoss_dWeightedSum;
            }
            currentLayer->dLoss_dWeightedSums->elements[outputNeuronIndex] = dLoss_dWeightedSum;
        }

        for(layerIndex = network->layerCount - 2; layerIndex >= 0; layerIndex --) {
            currentLayer = network->layers[layerIndex];
            for(int neuronIndex = 0; neuronIndex < currentLayer->neuronCount; neuronIndex++) {
                double dLoss_dOutput = 0.0f;
                
                Layer* nextLayer = network->layers[layerIndex + 1];
                for(int neuronIndexNext = 0; neuronIndexNext < nextLayer->neuronCount; neuronIndexNext++) {
                    dLoss_dOutput += nextLayer->dLoss_dWeightedSums->elements[neuronIndexNext] * nextLayer->weights->data[neuronIndexNext]->elements[neuronIndex];
                }
                double dOutput_dWeightedSum = currentLayer->activationFunction->derivative(currentLayer->weightedSums->elements[neuronIndex]);
                double dLoss_dWeightedSum = dLoss_dOutput * dOutput_dWeightedSum;

                for(int weightIndex = 0; weightIndex < currentLayer->weights->columns; weightIndex++) {
                    double dWeightedSum_dWeight = 0.0f;
                    if(layerIndex == 0) {
                        dWeightedSum_dWeight= network->layers[layerIndex]->input->elements[weightIndex];
                    }else {
                        dWeightedSum_dWeight = network->layers[layerIndex-1]->output->elements[weightIndex];
                    }
                    
                    double dLoss_dWeight = dLoss_dWeightedSum * dWeightedSum_dWeight;

                    currentLayer->gradients->data[neuronIndex]->elements[weightIndex] += dLoss_dWeight;
                }

                if(network->optimizationConfig->shouldUseGradientClipping == 1) {
                    if(dLoss_dWeightedSum < network->optimizationConfig->gradientClippingLowerBound) {
                        currentLayer->biasGradients->elements[neuronIndex] = network->optimizationConfig->gradientClippingLowerBound;
                    }else if(dLoss_dWeightedSum > network->optimizationConfig->gradientClippingUpperBound) {
                        currentLayer->biasGradients->elements[neuronIndex] = network->optimizationConfig->gradientClippingUpperBound;                        
                    }
                }else {
                    currentLayer->biasGradients->elements[neuronIndex] = dLoss_dWeightedSum;
                }

                currentLayer->dLoss_dWeightedSums->elements[neuronIndex] = dLoss_dOutput * dOutput_dWeightedSum;
            }
        }
    }
}

void deleteNNetwork(NNetwork* network){
    for(int i = network->layerCount - 1; i >= 0; i--) {
        deleteLayer(network->layers[i]);
    }
}

double accuracy(Matrix* targets, Matrix* outputs) {
   
    int value = 0;
    for(int i = 0; i < outputs->rows; i++){
        int predictedCategory = getIndexOfMax(outputs->data[i]);
        int targetCategory = getIndexOfMax(targets->data[i]);
   
        if (predictedCategory == targetCategory) {
            value++;
        }    
    }

    return (double)value / (double)outputs->rows;
}

void dumpNetworkState(NNetwork* network) {
   char json_output[5000]; // adjust the size to suit your needs

    sprintf(
        json_output,
        "{\n"
        "\t\"Network Config\": {\n"
        "\t\t\"Loss Function\": %s \n"
        "\t\t\"Layers\": [\n",
        "cross_entropy" // TODO: Turn into const values
    );

    for(int i = 0; i < network->layerCount; i++) {
        Layer* layer = network->layers[i];
        sprintf(
            json_output + strlen(json_output), // add to the end of the current string
            "\t\t\t{\n"
            "\t\t\t\t\"Layer Index\": %d,\n"
            "\t\t\t\t\"Number Of Neurons\": %d,\n"
            "\t\t\t\t\"Input Size\": %d,\n"
            "\t\t\t\t\"Weight Initialization Method\": \"%s\",\n"
            "\t\t\t\t\"Biases in Range\": [%.2f, %.2f]\n"
            "\t\t\t\t\"Activation Function\": %s \n" // TODO: Actually read this from the layer struct
            "\t\t\t}\n%s", // add comma for all but last layer
            i, layer->neuronCount, layer->weights->columns, "he initializer", -.5, .5, 
            i < network->layerCount - 1 ? "leaky relu" : "softmax",
            (i == network->layerCount - 1) ? "" : ",\n"
        );
    }

    sprintf(json_output + strlen(json_output), // add to the end of the current string
        "\t\t\t]\n"
        "\t}\n"
    );

    OptimizationConfig* opt_config = network->optimizationConfig/* get your OptimizationConfig object here */;

    sprintf(
        json_output + strlen(json_output),
        "\t\t\"Optimizer Config\": {\n"
        "\t\t\t\"shouldUseGradientClipping\": %d,\n"
        "\t\t\t\"gradientClippingLowerBound\": %.2f,\n"
        "\t\t\t\"gradientClippingUpperBound\": %.2f,\n"
        "\t\t\t\"shouldUseLearningRateDecay\": %d,\n"
        "\t\t\t\"learningRateDecayAmount\": %.2f,\n"
        "\t\t\t\"shouldUseMomentum\": %d,\n"
        "\t\t\t\"momentum\": %.2f,\n"
        "\t\t\t\"optimizer\": %s,\n"
        "\t\t\t\"epsilon\": %.2f,\n"
        "\t\t\t\"rho\": %.2f,\n"
        "\t\t\t\"beta1\": %.2f,\n"
        "\t\t\t\"beta2\": %.2f\n"
        "\t\t},\n", // add a comma as this might not be the last item in its parent object
        opt_config->shouldUseGradientClipping,
        opt_config->gradientClippingLowerBound,
        opt_config->gradientClippingUpperBound,
        opt_config->shouldUseLearningRateDecay,
        opt_config->learningRateDecayAmount,
        opt_config->shouldUseMomentum,
        opt_config->momentum,
        get_optimizer_name(opt_config->optimizer),
        opt_config->epsilon,
        opt_config->rho,
        opt_config->beta1,
        opt_config->beta2
    );


    // end of log
    // spritnf(json_output + strlen(json_output),
    //     "}\n"
    // );

    log_info("%s", json_output);

}