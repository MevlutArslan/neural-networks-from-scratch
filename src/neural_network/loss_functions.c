#include "loss_functions.h"

// double meanSquaredErrorDerivative(double error) {
//     return -2 * error;
// }


double meanSquaredError(Matrix* outputs, Vector* targets) {
    double mse = 0.0;

    for (int i = 0; i < outputs->rows; i++) {
        Vector* output = outputs->data[i];  
        double target = targets->elements[i];

        double difference = target - output->elements[0]; // assuming output vector is of size 1
        mse += (difference * difference) / 2;
    }

    mse /= (double)outputs->rows;

    return mse;
}


double meanSquaredErrorDerivative(double target, double predicted) {
    return -1 * (target - predicted);
}