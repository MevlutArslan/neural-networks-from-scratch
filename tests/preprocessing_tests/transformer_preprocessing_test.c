#include "transformer_preprocessing_test.h"

#define MAX_SEQ_LEN 10
#define VOCABULARY_SIZE 50

char* test_file_loc = "/Users/mevlutarslan/Documents/neural-networks-from-scratch/tests/preprocessing_tests/test_text.txt";

void line_to_embedding_test() {
    // Initialize your map_t and other required data structures
    map_t char_int_map = hashmap_new();

    hashmap_put(char_int_map, "n", 8);
    hashmap_put(char_int_map, ".", 10);
    hashmap_put(char_int_map, "e", 7);
    hashmap_put(char_int_map, "c", 9);
    hashmap_put(char_int_map, " ", 4);
    hashmap_put(char_int_map, "s", 3);
    hashmap_put(char_int_map, "T", 0);
    hashmap_put(char_int_map, "h", 1);
    hashmap_put(char_int_map, "a", 5);
    hashmap_put(char_int_map, "t", 6);
    hashmap_put(char_int_map, "i", 2);

    char text[26] = "This is a test sentence .";

    Matrix* sentence_embedding = line_to_embedding(text, 15, 15, char_int_map);

    // Perform assertions to check if the sentence_embedding is generated correctly
    // You can check dimensions, specific values, etc.
    assert(sentence_embedding->data[0]->elements[0] == 0);   // 'T'
    assert(sentence_embedding->data[0]->elements[1] == 1);   // 'h'
    assert(sentence_embedding->data[0]->elements[2] == 2);   // 'i'
    assert(sentence_embedding->data[0]->elements[3] == 3);   // 's'

    assert(sentence_embedding->data[1]->elements[0] == 4);   // ' '

    assert(sentence_embedding->data[2]->elements[0] == 2);   // 'i' 
    assert(sentence_embedding->data[2]->elements[1] == 3);   // 's' 

    assert(sentence_embedding->data[3]->elements[0] == 4);   // ' ' 

    assert(sentence_embedding->data[4]->elements[0] == 5);   // 'a' 

    assert(sentence_embedding->data[5]->elements[0] == 4);   // ' ' 

    assert(sentence_embedding->data[6]->elements[0] == 6);   // 't' 
    assert(sentence_embedding->data[6]->elements[1] == 7);   // 'e' 
    assert(sentence_embedding->data[6]->elements[2] == 3);   // 's' 
    assert(sentence_embedding->data[6]->elements[3] == 6);   // 't' 

    assert(sentence_embedding->data[7]->elements[0] == 4);   // ' ' 

    assert(sentence_embedding->data[8]->elements[0] == 3);   // 's' 
    assert(sentence_embedding->data[8]->elements[1] == 7);   // 'e' 
    assert(sentence_embedding->data[8]->elements[2] == 8);   // 'n' 
    assert(sentence_embedding->data[8]->elements[3] == 6);   // 't' 
    assert(sentence_embedding->data[8]->elements[4] == 7);   // 'e'
    assert(sentence_embedding->data[8]->elements[5] == 8);   // 'n' 
    assert(sentence_embedding->data[8]->elements[6] == 9);   // 'c' 
    assert(sentence_embedding->data[8]->elements[7] == 7);   // 'e' 

    assert(sentence_embedding->data[9]->elements[0] == 4);   // ' '

    assert(sentence_embedding->data[10]->elements[0] == 10); // '.' 

    // Clean up allocated memory
    hashmap_free(char_int_map);
    free_matrix(sentence_embedding);

    log_info("line_to_embedding test passed successfully.");
}

void word_to_embedding_test() {
    // Initialize your map_t and other required data structures
    map_t char_int_map = hashmap_new();

    hashmap_put(char_int_map, "n", 8);
    hashmap_put(char_int_map, ".", 10);
    hashmap_put(char_int_map, "e", 7);
    hashmap_put(char_int_map, "c", 9);
    hashmap_put(char_int_map, " ", 4);
    hashmap_put(char_int_map, "s", 3);
    hashmap_put(char_int_map, "T", 0);
    hashmap_put(char_int_map, "h", 1);
    hashmap_put(char_int_map, "a", 5);
    hashmap_put(char_int_map, "t", 6);
    hashmap_put(char_int_map, "i", 2);

    // Initialize your vector and other data structures
    char* token = "This";

    Vector* word_embedding = create_vector(100); // Replace with the appropriate size

    word_to_embedding(token, char_int_map, word_embedding);

    // Perform assertions to check if the vector is filled correctly
    assert(word_embedding->elements[0] == 0);   // 'T'
    assert(word_embedding->elements[1] == 1);   // 'h'
    assert(word_embedding->elements[2] == 2);   // 'i'
    assert(word_embedding->elements[3] == 3);   // 's'    
    
    // Clean up allocated memory
    hashmap_free(char_int_map);
    free_vector(word_embedding);

    log_info("word_embedding test passed successfully.");
}

void empty_word_to_embedding_test() {
    // Initialize your map_t and other required data structures
    map_t char_int_map = hashmap_new();

    hashmap_put(char_int_map, "n", 8);
    hashmap_put(char_int_map, ".", 10);
    hashmap_put(char_int_map, "e", 7);
    hashmap_put(char_int_map, "c", 9);
    hashmap_put(char_int_map, " ", 4);
    hashmap_put(char_int_map, "s", 3);
    hashmap_put(char_int_map, "T", 0);
    hashmap_put(char_int_map, "h", 1);
    hashmap_put(char_int_map, "a", 5);
    hashmap_put(char_int_map, "t", 6);
    hashmap_put(char_int_map, "i", 2);

    // Initialize your vector and other data structures
    char* token = " ";

    Vector* word_embedding = create_vector(100); 

    word_to_embedding(token, char_int_map, word_embedding);

    // Perform assertions to check if the vector is filled correctly
    assert(word_embedding->elements[0] == 4);   // ' '
    
    // Clean up allocated memory
    hashmap_free(char_int_map);
    free_vector(word_embedding);

    log_info("word_embedding with space character test passed successfully.");
}

void fill_tokenizer_vocabulary_test() {
    // Initialize your map_t and other required data structures
    map_t expected_char_int_map = hashmap_new();
    map_t result_char_int_map = hashmap_new();
    int index[1] = {0};
    
    // Initialize your text and other data structures
    hashmap_put(expected_char_int_map, "T", 0);
    hashmap_put(expected_char_int_map, "h", 1);
    hashmap_put(expected_char_int_map, "i", 2);
    hashmap_put(expected_char_int_map, "s", 3);
    hashmap_put(expected_char_int_map, " ", 4);
    hashmap_put(expected_char_int_map, ".", 5);

    char* text = "This .";

    fill_tokenizer_vocabulary(text, result_char_int_map, NULL, &index);

    // Perform assertions to check if the vocabulary is filled correctly
    assert(hashmap_length(expected_char_int_map) == hashmap_length(result_char_int_map));

    any_t expected_return;
    any_t actual_return;

    hashmap_get(expected_char_int_map, "T", &expected_return);
    hashmap_get(expected_char_int_map, "T", &actual_return);
    assert((int) expected_return == (int) actual_return);

    hashmap_get(expected_char_int_map, "h", &expected_return);
    hashmap_get(expected_char_int_map, "h", &actual_return);
    assert((int) expected_return == (int) actual_return);


    hashmap_get(expected_char_int_map, "i", &expected_return);
    hashmap_get(expected_char_int_map, "i", &actual_return);
    assert((int) expected_return == (int) actual_return);


    hashmap_get(expected_char_int_map, "s", &expected_return);
    hashmap_get(expected_char_int_map, "s", &actual_return);
    assert((int) expected_return == (int) actual_return);


    hashmap_get(expected_char_int_map, " ", &expected_return);
    hashmap_get(expected_char_int_map, " ", &actual_return);
    assert((int) expected_return == (int) actual_return);
    
    hashmap_get(expected_char_int_map, ".", &expected_return);
    hashmap_get(expected_char_int_map, ".", &actual_return);
    assert((int) expected_return == (int) actual_return);
    
    // Clean up allocated memory
    hashmap_free(expected_char_int_map);
    hashmap_free(result_char_int_map);

    log_info("fill_tokenizer_vocabulary test passed successfully.");
}

void add_positional_embeddings_test() {
    map_t char_int_map = hashmap_new();
    map_t int_char_map = hashmap_new();

    MatrixArray* embeddings = load_text_as_embedding(test_file_loc, char_int_map, int_char_map, 20, 20);
    MatrixArray* embeddings_copy = create_matrix_arr(embeddings->length);

    for(int i = 0; i < embeddings->length; i++) {
        embeddings_copy->array[i] = copy_matrix(embeddings->array[i]);
    }

    add_positional_embeddings(embeddings);

    // assertions to check if it has been updated correctly
    for(int i = 0; i < embeddings->length; i++) {
        assert(is_equal_matrix(embeddings->array[i], embeddings_copy->array[i]) == FALSE);
    }

    log_info("add_positional_embeddings test passed successfully.");
}

void get_positional_embeddings_test() {
    int d_model = 3;
    Matrix* positional_embeddings = get_positional_embeddings(NULL, 2, d_model);

    // according to the results of a library i used to calculate positional encoding for a 2x3 embedding
    Matrix* expected_embeddings = create_matrix(2, 3);
    expected_embeddings->data[0]->elements[0] = 0.0;
    expected_embeddings->data[0]->elements[1] = 1.0;
    expected_embeddings->data[0]->elements[2] = 0.0;
    expected_embeddings->data[1]->elements[0] = 0.841470985;
    expected_embeddings->data[1]->elements[1] = 0.999997679;
    expected_embeddings->data[1]->elements[2] = 0.00000464158883;

    // Perform assertions to check if the calculated positional embeddings match the expected values
    assert(is_equal_matrix(positional_embeddings, expected_embeddings) == TRUE);
    
    // Clean up allocated memory
    free_matrix(positional_embeddings);
    free_matrix(expected_embeddings);

    log_info("get_positional_embeddings test passed successfully.");
}