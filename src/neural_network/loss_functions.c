#include "loss_functions.h"

double derivativeMeanSquaredError(double output, double target) {
    return -2 * (output - target);
}

double meanSquaredError(Matrix* outputs, Vector* targets) {
    double mse = 0.0;

    for (int i = 0; i < outputs->rows; i++) {
        Vector* output = outputs->data[i];  
        double target = targets->elements[i];

        double difference = output->elements[0] - target; // assuming output vector is of size 1
        mse += difference * difference;
    }

    mse /= outputs->rows;
    return mse;
}
