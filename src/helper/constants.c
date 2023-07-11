#include "constants.h"

const char* get_optimizer_name(int optimizer) {
    switch (optimizer) {
        case SGD:
            return "SGD";
        case ADAGRAD:
            return "ADAGRAD";
        case RMS_PROP:
            return "RMS_PROP";
        case ADAM:
            return "ADAM";
        default:
            return "UNKNOWN";
    }
}
