#include "transformer_preprocessing_test.h"

#define MAX_SEQ_LEN 10
#define VOCABULARY_SIZE 50

char* test_file_local = "/Users/mevlutarslan/Documents/neural-networks-from-scratch/tests/preprocessing_tests/test_text.txt";

void test_line_to_embedding() {
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

void test_word_to_embedding() {
    // Initialize your map_t and other required data structures
    map_t char_int_map;
    // Initialize your vector and other data structures

    char* token = "word";

    Vector* vector = create_vector(100); // Replace with the appropriate size

    word_to_embedding(token, char_int_map, vector);

    // Perform assertions to check if the vector is filled correctly

    // Clean up allocated memory

}

void test_fill_tokenizer_vocabulary() {
    // Initialize your map_t and other required data structures
    map_t char_int_map;
    map_t int_char_map;
    int index = 0;
    // Initialize your text and other data structures

    char* text = "This is a test.";

    fill_tokenizer_vocabulary(text, char_int_map, int_char_map, &index);

    // Perform assertions to check if the vocabulary is filled correctly

    // Clean up allocated memory

}