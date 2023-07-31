#include "math_tasks.h"
#include "nmath.h"
#include "nmatrix.h"
#include <stdlib.h>
#include <sys/types.h>

struct MatrixVectorOperation* create_matrix_vector_operation(const Matrix* matrix, const Vector* vector, Vector* output, int begin_index, int end_index) {
    struct MatrixVectorOperation* data = (struct MatrixVectorOperation*) malloc(sizeof(struct MatrixVectorOperation));
    data->matrix = matrix;
    data->vector = vector;
    data->output = output;
    data->begin_index = begin_index;
    data->end_index = end_index;
    
    return data;
}

struct MatrixVectorOperation* copy_matrix_vector_operation(const struct MatrixVectorOperation* original, int begin_index, int end_index) {
    struct MatrixVectorOperation* copy = malloc(sizeof(struct MatrixVectorOperation));
    *copy = *original;

    copy->begin_index = begin_index;
    copy->end_index = end_index;

    return copy;
}

void push_dot_product_as_task(struct ThreadPool* thread_pool, struct MatrixVectorOperation* data) {
    assert(data->matrix->columns == data->vector->size);
    if(data->end_index - data->begin_index <= MIN_ELEMENTS_PER_THREAD) {
        struct Task* task = create_task(parallelized_dot_product, data);
        if(thread_pool->push_task == NULL) {
            log_error("ERROR WITH PUSH TASK");
        }
        thread_pool->push_task(thread_pool, task);
    }else {
        // divide and conquer
        // mid = low + ((high - low) / 2)
        int middle_index = data->begin_index + (data->end_index - data->begin_index) / 2;

        struct MatrixVectorOperation* copy_one = copy_matrix_vector_operation(data, data->begin_index, middle_index);
        push_dot_product_as_task(thread_pool, copy_one);

        struct MatrixVectorOperation* copy_two = copy_matrix_vector_operation(data, middle_index, data->end_index);
        push_dot_product_as_task(thread_pool, copy_two);
    }
}