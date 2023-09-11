#include "loss_functions.h"

double mean_squared_error(Matrix* targets, Matrix* outputs) {
    double mse = 0.0;

    for (int i = 0; i < outputs->rows; i++) {
        Vector* output = outputs->data[i];
        Vector* target = targets->data[i];

        // assuming both vectors have size 1 because of the nature of MSE use cases.
        double difference = target->elements[0] - output->elements[0];
        /* 
            Dividing by 2 in the loss calculation ensures that the derivative simplifies to (predicted - target).
            This simplification aids in gradient computation. 
        */
        mse += (difference * difference) / 2;


        #ifdef DEBUG
            char* output_vector_str = vector_to_string(output);
            char* target_vector_str = vector_to_string(target);
            
            log_debug(
                "MSE Input & Intermediate Output: \n"
                "Output Vector: %s \n"
                "Target Vector: %s \n"
                "Squared Difference: %f \n", 
                output_vector_str, 
                target_vector_str, 
                difference * difference / 2
            );
            
            free(output_vector_str);
            free(target_vector_str);
        #endif
    }

    mse /= (double)outputs->rows;

    #ifdef DEBUG
        log_debug("Final MSE: %f \n", mse);
    #endif

    return mse;
}

double mean_squared_error_derivative(double target, double predicted) {
    return -1 * (target - predicted);
}

Matrix* mean_squared_error_derivative_batched(Matrix* target, Matrix* output) {
    Matrix* derivatives = create_matrix(output->rows, output->columns);

    for(int i = 0; i < output->rows; i++) {
        for(int j = 0; j < output->columns; j++) {
            derivatives->data[i]->elements[j] = mean_squared_error_derivative(target->data[i]->elements[j], output->data[i]->elements[j]);
        }
    }

    return derivatives;
}

/*
    Should be used only when both target and prediction are one hot encoded vectors!
    The formula for Categorical Cross Entropy simplifies to: 
        -1 * log(prediction_vector[index of the target category in the target vector])
*/
double categorical_cross_entropy_loss_per_row(Vector* target, Vector* output) {
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
double categorical_cross_entropy_loss(Matrix* targetOutputs, Matrix* outputs) {
    double sum = 0.0f;
    for(int i = 0; i < outputs->rows; i++) {
        Vector* targetVector = targetOutputs->data[i];
        Vector* outputVector = outputs->data[i];
        
        sum += categorical_cross_entropy_loss_per_row(targetVector, outputVector);
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
Vector* categorical_cross_entropy_loss_derivative(Vector* target, Vector* predictions) {
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

void categorical_cross_entropy_loss_derivative_batched(Matrix* target, Matrix* prediction, Matrix* loss_wrt_output) {
    for(int i = 0; i < target->rows; i++) {
        Vector* derivatives = categorical_cross_entropy_loss_derivative(target->data[i], prediction->data[i]);
        assert(loss_wrt_output->data[i]->size == derivatives->size);
        memcpy(loss_wrt_output->data[i]->elements, derivatives->elements, derivatives->size * sizeof(double));
        free_vector(derivatives);
    }
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