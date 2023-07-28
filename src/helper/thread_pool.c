#include "thread_pool.h"
#include <stdlib.h>
#include "../../libraries/logger/log.h"

TaskQueue* init_task_queue(int capacity) {
    TaskQueue* t_queue = malloc(sizeof(TaskQueue));
    t_queue->capacity = capacity;
    t_queue->number_of_elements = 0;
    t_queue->tasks = calloc(t_queue->capacity, sizeof(Task));

    return t_queue;
}

int add_to_task_queue(TaskQueue* queue, Task* task) {
    if(queue->number_of_elements >= queue->capacity) {
        return QUEUE_FULL_ERRROR;
    }

    queue->tasks[queue->number_of_elements] = task;
    queue->number_of_elements++;

    return QUEUE_ADD_SUCCESS;
}

Task* pop_from_task_queue(TaskQueue* queue) {
    if(queue->number_of_elements == 0) {
        log_error("Task Queue is empty!");
        return NULL;
    }

    Task* popped = queue->tasks[0];

    for(int i = 0; i < queue->number_of_elements - 1; i++) {
        queue->tasks[i] = queue->tasks[i + 1];
    }

    return popped;
}