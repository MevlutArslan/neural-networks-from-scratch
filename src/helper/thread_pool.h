#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <pthread.h>

#define NUMBER_OF_THREADS 8
#define MAX_NUMBER_OF_TASKS 256

#define QUEUE_ADD_SUCCESS 1
#define QUEUE_FULL_ERRROR -1
#define QUEUE_IS_EMPTY_ERROR -2

pthread_t threads[NUMBER_OF_THREADS];

typedef struct {
    // function pointer
    // task type
} Task;

// If you need more performance change the data structure to be circular buffer instead of queue.
typedef struct {
    Task** tasks;
    int capacity;
    int number_of_elements;
}TaskQueue;

TaskQueue* init_task_queue(int capacity);
int add_to_task_queue(TaskQueue* queue, Task* task);
Task* pop_from_task_queue(TaskQueue* queue);


#endif