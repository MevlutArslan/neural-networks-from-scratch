#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "../tests/test.h"
#include "../libraries/logger/log.h"
#include "helper/thread_pool.h"
#include "nmath/nmath.h"
#include "networks/model.h"
#include "helper/hashmap.h"
void runProgram();

#define TIME_BUFFER_SIZE 64

int main(int argc, char* argv[])
{
    int isTesting = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "test") == 0) {
            isTesting = 1;
            break;
        }
    }

    if (isTesting) {
        log_info("%s", "Running tests...");
        run_tests();
    } else {
        log_info("%s", "Running Program!");
        runProgram();   
    }
}



void runProgram() {
    srand(306);

    // Model* model = create_wine_categorization_model();
    // clock_t start = clock();

    // train_model(model, FALSE);
    // // validate_model(model);
    
    // clock_t end = clock();

    // log_info("time it took to process mnist: %f",  (((double) (end - start)) / CLOCKS_PER_SEC) * 1000);
    // free_model(model);

    map_t char_int_map = hashmap_new();
    map_t int_char_map = hashmap_new();

    // create matrices for each max length sequence. (from my preprocessing done in python: longest sentence is 400 words, 2200 chars long and vocab size is 80)
    int d_model = 80;
    int max_seq_len = 400;
    MatrixArray* embeddings = load_text_as_embedding("/Users/mevlutarslan/Downloads/datasets/paul_gram_essays.txt", char_int_map, int_char_map, 400, d_model);
    
    add_positional_embeddings(embeddings);

    // create query, key and value vectors per word
    MatrixArray* query_weights = create_matrix_arr(embeddings->length);
    MatrixArray* key_weights = create_matrix_arr(embeddings->length);
    MatrixArray* value_weights = create_matrix_arr(embeddings->length);

    for(int i = 0; i < embeddings->length; i++) {
        query_weights->array[i] = he_initialize_matrix(max_seq_len, d_model);
        key_weights->array[i] = he_initialize_matrix(max_seq_len, d_model);
        value_weights->array[i] = he_initialize_matrix(max_seq_len, d_model);
    }

    MatrixArray* queries = create_matrix_arr(embeddings->length);
    MatrixArray* keys = create_matrix_arr(embeddings->length);
    MatrixArray* values = create_matrix_arr(embeddings->length);

    for(int i = 0; i < embeddings->length; i++) {
        queries->array[i] = matrix_multiplication(embeddings->array[i], query_weights->array[i]);
        keys->array[i] = matrix_multiplication(embeddings->array[i], key_weights->array[i]);
        values->array[i] = matrix_multiplication(embeddings->array[i], value_weights->array[i]);
    }


    // Clean up
    hashmap_free(char_int_map);
    hashmap_free(int_char_map);

    free_matrix_arr(embeddings);

    free_matrix_arr(query_weights);
    free_matrix_arr(key_weights);
    free_matrix_arr(value_weights);

    free_matrix_arr(queries);
    free_matrix_arr(keys);
    free_matrix_arr(values);

    // printf("All tests passed\n");
}   
