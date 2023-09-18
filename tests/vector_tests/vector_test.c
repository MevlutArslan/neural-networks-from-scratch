#include "vector_test.h"

void split_vector_test() {
    Vector* input = create_vector(10);

    int num_vectors = 5;
    int sub_vector_len = 2;

    VectorArray* vectors = split_vector(input, num_vectors);

    assert(vectors->length == num_vectors);
    
    for(int i = 0; i < vectors->length; i++) {
        Vector* sub_vector = vectors->vectors[i];
        assert(sub_vector->size == sub_vector_len);
        
        for(int j = 0; j < sub_vector->size; j++) {
            assert(sub_vector->elements[j] == input->elements[(i * sub_vector_len) + j]);
        }
    }

    free_vector_arr(vectors);
    free_vector(input);
    
    log_info("split_vector test passed successfully.");
}

void concatenate_vectors_test() {
    VectorArray* vectors = create_vector_arr(3);
    
    vectors->vectors[0] = create_vector(2);
    vectors->vectors[1] = create_vector(2);
    vectors->vectors[2] = create_vector(2);

    // Fill the vectors with some sample data
    for (int i = 0; i < 2; i++) {
        vectors->vectors[0]->elements[i] = i;
    }

    for (int i = 0; i < 2; i++) {
        vectors->vectors[1]->elements[i] = i + 2;
    }

    for (int i = 0; i < 2; i++) {
        vectors->vectors[2]->elements[i] = i + 4;
    }

    Vector* concatenated_vector = concatenate_vectors(vectors);
    assert(concatenated_vector->size == 2 + 2 + 2);

    for (int i = 0; i < concatenated_vector->size; i++) {
        assert(concatenated_vector->elements[i] == i);
    }

    free_vector_arr(vectors);
    free_vector(concatenated_vector);

    log_info("concatenate_vectors_test passed successfully.");
}
