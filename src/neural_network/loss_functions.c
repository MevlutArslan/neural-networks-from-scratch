#include "loss_functions.h"

double meanSquaredError(Matrix* outputs, Matrix* targets) {
    double mse = 0.0;

    for (int i = 0; i < outputs->rows; i++) {
        Vector* output = outputs->data[i];
        Vector* target = targets->data[i];

        double difference = target->elements[0] - output->elements[0]; // assuming both vectors have size 1
        mse += (difference * difference) / 2;

        if(DEBUG == 1) {
            char* outputVectorStr = vectorToString(output);
            char* targetVectorStr = vectorToString(target);
            
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
        }
    }

    mse /= (double)outputs->rows;

    if(DEBUG == 1) {
        log_debug("Final MSE: %f \n", mse);
    }

    return mse;
}


double meanSquaredErrorDerivative(double target, double predicted) {
    return -1 * (target - predicted);
}

double crossEntropyForEachRow(Vector* target, Vector* output) {
    double sum = 0.0;
    int size = target->size;

    for (int i = 0; i < size; i++) {
        sum += -1 * (target->elements[i]) * log(output->elements[i]);
    }
    
    if(DEBUG == 1) {
        char* targetVectorStr = vectorToString(target);
        char* outputVectorStr = vectorToString(output);

        // log the inputs and the output
        log_debug(
            "Cross Entropy Inputs & Output: \n"
            "\t\t\t\tTarget Vector: %s \n"
            "\t\t\t\tOutput Vector: %s \n"
            "\t\t\t\tCross Entropy: %f \n", 
            targetVectorStr, 
            outputVectorStr, 
            sum
        );

        free(targetVectorStr);
        free(outputVectorStr);
    }
    return sum;
}


double crossEntropyLoss(Matrix* targetOutputs, Matrix* outputs) {
    double totalLoss = 0.0;
    int dataSize = ROWS_TO_USE;
    for (int i = 0; i < dataSize; i++) {
        Vector* target = targetOutputs->data[i];
        Vector* output = outputs->data[i];
        totalLoss += crossEntropyForEachRow(target, output);
    }
    if(DEBUG == 1) {
        log_info("Called crossEntropyLoss & returned: %f \n", totalLoss / ROWS_TO_USE);
    }
    return totalLoss / ROWS_TO_USE;
}

double crossEntropyLossDerivative(double target, double predicted) {
    if(DEBUG == 1) {
        log_info("Called crossEntropyLossDerivative & returned: %f \n", predicted - target);
    }
    return (predicted - target);
}