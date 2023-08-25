#include "nvector.h"

Vector* create_vector(int size) {
    Vector* vector = malloc(sizeof(Vector));
    vector->elements = calloc(size, sizeof(double));
    vector->size = size;
    
    return vector;
}

Vector** create_vector_arr(int length) {
    Vector** array = (Vector**) malloc(length * sizeof(Vector*));

    for(int i = 0; i < length; i++) {
        array[i] = NULL;
    }

    return array;
}

void free_vector(Vector* vector) {
    if(vector == NULL) {
        return;
    }
    free(vector->elements);
    free(vector);
}

void fill_vector_random(Vector* vector, double min, double max) {
    for(int i = 0; i < vector->size; i++) {
        vector->elements[i] = ((double)rand() / (double)RAND_MAX) * (max - min) + min;
    }
}

void fill_vector(Vector* vector, double value) {
    for(int i = 0; i < vector->size; i++) {
        vector->elements[i] = value;
    }
}

char* vector_to_string(const Vector* vector) {
    char* output = (char*) malloc(sizeof(char) * vector->size * 20); // Assume up to 20 characters per number
    if (output == NULL) {
        // handle error
        return NULL;
    }

    sprintf(output, "[");
    int i;
    for(i = 0; i < vector->size - 1; i++) {
        char buffer[20];
        sprintf(buffer, "%f, ", vector->elements[i]);
        strcat(output, buffer);
    }
    
    // Add last element and closing bracket
    char buffer[20];
    sprintf(buffer, "%f]", vector->elements[i]);
    strcat(output, buffer);

    return output;
}


Vector* copy_vector(const Vector* vector) {
    Vector* copy = create_vector(vector->size);

    for(int i = 0; i < vector->size; i++) {
        copy->elements[i] = vector->elements[i];
    }
    
    return copy;
}

Vector* slice_vector(const Vector* vector, int beginIndex, int endIndex) {
    if(endIndex > vector->size) {
        log_error("End index #%d is out of bounds!", endIndex);
        return NULL;
    }

    int newSize = endIndex - beginIndex;
    Vector* newVector = create_vector(newSize);

    for (int i = beginIndex, j = 0; i < endIndex; i++, j++) {
        newVector->elements[j] = vector->elements[i];
    }

    return newVector;
}

Vector* array_to_vector(double* array, int length) {
    Vector* vector = create_vector(length);

    for(int i = 0; i < vector->size; i++) {
        vector->elements[i] = array[i];
    }

    return vector;
}

char* serialize_vector(const Vector* vector) {
    cJSON *root = cJSON_CreateObject();
    cJSON *elements = cJSON_CreateArray();

    for (int i = 0; i < vector->size; i++) {
        cJSON_AddItemToArray(elements, cJSON_CreateNumber(vector->elements[i]));
    }

    cJSON_AddItemToObject(root, "size", cJSON_CreateNumber(vector->size));
    cJSON_AddItemToObject(root, "elements", elements);

    char *jsonString = cJSON_PrintUnformatted(root);

    cJSON_Delete(root);

    return jsonString;
}

Vector* deserialize_vector(cJSON* json) {
    Vector* vector = malloc(sizeof(Vector));
    vector->size = cJSON_GetObjectItem(json, "size")->valueint;
    vector->elements = calloc(vector->size,  sizeof(double));

    cJSON* array = cJSON_GetObjectItem(json, "elements");
    for(int i = 0; i < vector->size; i++) {
        double value = cJSON_GetArrayItem(array, i)->valuedouble;
        vector->elements[i] = value;
    }
    return vector;
}