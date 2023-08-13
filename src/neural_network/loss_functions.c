#include "loss_functions.h"

LossFunction MEAN_SQUARED_ERROR = { .loss_function = meanSquaredError, .derivative = meanSquaredErrorDerivative, .name = "MEAN_SQUARED_ERROR" };
LossFunction CATEGORICAL_CROSS_ENTROPY = { .loss_function = categoricalCrossEntropyLoss, .name = "CATEGORICAL_CROSS_ENTROPY" };

double meanSquaredError(Matrix* outputs, Matrix* targets) {
    double mse = 0.0;

    for (int i = 0; i < outputs->rows; i++) {
        Vector* output = outputs->data[i];
        Vector* target = targets->data[i];

        double difference = target->elements[0] - output->elements[0]; // assuming both vectors have size 1
        mse += (difference * difference) / 2;

        #ifdef DEBUG
            char* outputVectorStr = vector_to_string(output);
            char* targetVectorStr = vector_to_string(target);
            
            log_debug(
                "MSE Input & Intermediate Output: \n"
                "Output Vector: %s \n"
                "Target Vector: %s \n"
                "Squared Difference: %f \n", 
                outputVectorStr, 
                targetVectorStr, 
                difference * difference / 2
            );
            
            free(outputVectorStr);
            free(targetVectorStr);
        #endif
    }

    mse /= (double)outputs->rows;

    #ifdef DEBUG
        log_debug("Final MSE: %f \n", mse);
    #endif

    return mse;
}


double meanSquaredErrorDerivative(double target, double predicted) {
    return -1 * (target - predicted);
}

/*
    Should be used when both target and prediction are one hot encoded vectors!
    The formula for Categorical Cross Entropy simplifies to: 
        -1 * log(prediction_vector[index of the target category in the target vector])
*/
double categoricalCrossEntropyPerInput(Vector* target, Vector* output) {
    // natural log
    double loss = -1 * target->elements[arg_max(target)] * log(output->elements[arg_max(output)]);

    #ifdef DEBUG
        char* targetVectorStr = vector_to_string(target);
        char* outputVectorStr = vector_to_string(output);

        // log the inputs and the output
        log_debug(
            "Cross Entropy Inputs & Output: \n"
            "\t\t\t\tTarget Vector: %s \n"
            "\t\t\t\tOutput Vector: %s \n"
            "\t\t\t\tCross Entropy: %f \n", 
            targetVectorStr, 
            outputVectorStr, 
            loss
        );

        free(targetVectorStr);
        free(outputVectorStr);
    #endif
    // we don't care about the cases where target_i is 0 as 0 * log(output_i) will return 0 nonetheless
    return loss;
}

/* 
    * The categorical cross-entropy loss is exclusively used in multi-class classification tasks.
*/
double categoricalCrossEntropyLoss(Matrix* targetOutputs, Matrix* outputs) {
    double sum = 0.0f;
    for(int i = 0; i < outputs->rows; i++) {
        Vector* targetVector = targetOutputs->data[i];
        Vector* outputVector = outputs->data[i];

        sum += categoricalCrossEntropyPerInput(targetVector, outputVector);
    }
    #ifdef DEBUG
        // todo: change this back to output.rows
        log_info("%s", "Called multiClassCrossEntropyLoss & returned: %f \n", sum / 20);
    #endif
    return sum / outputs->rows;
}

/*
    * Derivative of sum equals sum of the derivatives
    * Derivative of the logarithmic function is the reciprocal of its parameter, multiplied by the partial derivative of this parameter.
      1/prediction_vector[index] * d_prediction_vector[index]
*/
Vector* categoricalCrossEntropyLossDerivative(Vector* target, Vector* predictions) {
    Vector* lossGrads = create_vector(predictions->size);
    for(int i = 0; i < predictions->size; i++) {
        double target_i = target->elements[i];
        double prediction_i = predictions->elements[i];
        lossGrads->elements[i] = -1 * (target_i / prediction_i + 1e-7);
    }

    #ifdef DEBUG
        char* targetVectorStr = vector_to_string(target);
        char* outputVectorStr = vector_to_string(predictions);
        char* lossGradStr = vector_to_string(lossGrads);
        log_debug(
            "Cross Entropy Derivative Inputs & Output: \n"
            "\t\t\t\tTarget Vector: %s \n"
            "\t\t\t\tOutput Vector: %s \n"
            "\t\t\t\tLoss Grads: %s \n", 
            targetVectorStr, 
            outputVectorStr, 
            lossGradStr
        );
    #endif

    return lossGrads;
}

const char* get_loss_function_name(const LossFunction* lossFunction) {
    return lossFunction->name;
}

void computeCategoricalCrossEntropyLossDerivativeMatrix(Matrix* target, Matrix* prediction, Matrix* loss_wrt_output) {
    for(int i = 0; i < target->rows; i++) {
        loss_wrt_output->data[i] = categoricalCrossEntropyLossDerivative(target->data[i], prediction->data[i]);
    }
}