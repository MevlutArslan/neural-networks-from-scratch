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