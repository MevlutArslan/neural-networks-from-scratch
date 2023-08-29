#include "loss_functions.h"
#include <string.h>

double meanSquaredError(Matrix* outputs, Matrix* targets) {
    double mse = 0.0;

    for (int i = 0; i < outputs->rows; i++) {
        double output = outputs->get_element(outputs, i, 0); // assuming both matrices have 1 column
        double target = targets->get_element(targets, i, 0);

        double difference = target - output;
        mse += (difference * difference) / 2;

        #ifdef DEBUG
            log_debug(
                "MSE Input & Intermediate Output: \n"
                "Output Value: %f \n"
                "Target Value: %f \n"
                "Squared Difference: %f \n", 
                output, 
                target, 
                difference * difference / 2
            );
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
double calculateCategoricalCrossEntropyPerInput(Matrix* target, int target_row, Matrix* output, int output_row) {
    // natural log
    int target_arg_max = arg_max_matrix_row(target, target_row);
    int output_arg_max = arg_max_matrix_row(output, output_row);

    double loss = -1 * target->get_element(target, target_row, target_arg_max) * log(output->get_element(output, output_row, output_arg_max));

    #ifdef DEBUG
        // log the inputs and the output
        log_debug(
            "Cross Entropy Inputs & Output: \n"
            "\t\t\t\tTarget max element index: %d \n"
            "\t\t\t\tOutput max element index: %d \n"
            "\t\t\t\tCross Entropy: %f \n", 
            target_arg_max, 
            output_arg_max, 
            loss
        );
    #endif
    // we don't care about the cases where target_i is 0 as 0 * log(output_i) will return 0 nonetheless
    return loss;
}


/* 
    * The categorical cross-entropy loss is exclusively used in multi-class classification tasks.
*/
double calculateCategoricalCrossEntropyLoss(Matrix* targetOutputs, Matrix* outputs) {
    double sum = 0.0f;
    for(int i = 0; i < outputs->rows; i++) {
        sum += calculateCategoricalCrossEntropyPerInput(targetOutputs, i, outputs, i);
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
Vector* calculateCategoricalCrossEntropyLossDerivative(Vector* target, Vector* predictions) {
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

void computeCategoricalCrossEntropyLossDerivativeMatrix(Matrix* target, Matrix* prediction, Matrix* loss_wrt_output) {
    Vector* target_row = create_vector(target->columns);
    Vector* prediction_row = create_vector(prediction->columns);

    for(int i = 0; i < target->rows; i++) {
        memcpy(target_row->elements, target->data->elements + (i * target->columns), target->columns * sizeof(double));
        memcpy(prediction_row->elements, prediction->data->elements + (i * prediction->columns), prediction->columns * sizeof(double));
        
        Vector* derivatives = calculateCategoricalCrossEntropyLossDerivative(target_row, prediction_row);
        assert(loss_wrt_output->columns == derivatives->size);
        
        memcpy(loss_wrt_output->data->elements + (i * loss_wrt_output->columns), derivatives->elements, derivatives->size * sizeof(double));
        free_vector(derivatives);
    }

    free_vector(target_row);
    free_vector(prediction_row);
}

LossFunctionType get_loss_fn_by_name(char* name) {
    if(strcmp(name, MEAN_SQUARED_ERROR_STR) == 0) {
        return MEAN_SQUARED_ERROR;
    }else if(strcmp(name, CATEGORICAL_CROSS_ENTROPY_STR) == 0) {
        return CATEGORICAL_CROSS_ENTROPY;
    }else {
        log_error("Unrecognized loss function name: %s", name);
        return UNRECOGNIZED_LFN;
    }
}

const char* loss_fn_to_string(const LossFunctionType lossFunction) {
    switch (lossFunction) {
        case MEAN_SQUARED_ERROR:
            return MEAN_SQUARED_ERROR_STR;
        case CATEGORICAL_CROSS_ENTROPY:
            return CATEGORICAL_CROSS_ENTROPY_STR;
        default:
            return "unrecognized_lfn";
    }
    
}