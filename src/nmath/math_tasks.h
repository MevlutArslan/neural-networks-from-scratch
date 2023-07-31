#ifndef MATH_TASKS_H
#define MATH_TASKS_H

#include "../helper/thread_pool.h"
#include "nmatrix.h"
#include "assert.h"
#include "nvector.h"

#define MIN_ELEMENTS_PER_THREAD 500

// typedef struct {
//     const Matrix* m1;
//     const Matrix* m2;
//     Matrix* output;
// } MatrixMatrixOperation;

struct MatrixVectorOperation {
    const Matrix* matrix;
    const Vector* vector;
    Vector* output;
    int begin_index;
    int end_index;
};

struct MatrixVectorOperation* create_matrix_vector_operation(const Matrix* matrix, const Vector* vector, Vector* output, int begin_index, int end_index);
struct MatrixVectorOperation* copy_matrix_vector_operation(const struct MatrixVectorOperation* original, int begin_index, int end_index);
// typedef struct {
//     const Vector* v1;
//     const Vector* v2;
//     Vector* output;
// } VectorVectorOperation;

// // Matrix <-> Matrix Operations
// void create_matrix_product_task(MatrixMatrixOperation* data);
// Task* create_matrix_addition_task(MatrixMatrixOperation* data);
// Task* create_matrix_subtraction_task(MatrixMatrixOperation* data);
// Task* create_matrix_multiplication_task(MatrixMatrixOperation* data);

// // Matrix <-> Vector Operations
void push_dot_product_as_task(struct ThreadPool* thread_pool, struct MatrixVectorOperation* data);

// // Vector <-> Vector Operations
// Task* create_vector_product_task(VectorVectorOperation* data);
// Task* create_vector_addition_task(VectorVectorOperation* data);
// Task* create_vector_subtraction_task(VectorVectorOperation* data);
// Task* create_vector_multiplication_task(VectorVectorOperation* data);

#endif