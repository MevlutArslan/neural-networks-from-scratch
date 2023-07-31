#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include "../../libraries/logger/log.h"

struct TaskQueue;
struct Task;

struct ThreadPool {
    pthread_t* threads;
    struct TaskQueue* task_queue;
    
    int num_threads;
    volatile int is_active; // flag to start/stop the threads

    int num_pending_tasks;
    int num_threads_exited;

    pthread_mutex_t lock; // lock for accessing the queue's element count.
    pthread_mutex_t capacity_lock; // lock for accesing queue's capacity.
    pthread_mutex_t pending_tasks_lock; // lock for accessing num_pending_tasks.

    pthread_cond_t signal; // cond for signaling there are tasks to handle.
    pthread_cond_t capacity_signal; // cond for signaling the queue has available space for new tasks.
    pthread_cond_t all_tasks_done; // cond for signaling the thread pool is done with all of its tasks.
    
    void (*push_task)(struct ThreadPool*, struct Task*);
    struct Task* (*pop_task)(struct ThreadPool*);
};

// allocates memory for the thread-pool and initializes properties
struct ThreadPool* create_thread_pool(int num_threads);
void destroy_thread_pool(struct ThreadPool* thread_pool);

void* run_thread(void* args);
void wait_for_all_tasks(struct ThreadPool* thread_pool);

struct TaskQueue {
    struct Task** tasks; // An array of pointers to tasks
    int capacity; // maximum number of elements the queue can hold.
    int element_count; // number of elements in the queue.
};

struct TaskQueue* init_task_queue(int capacity);
struct Task {
    void(*function)(void*); // pointer to the function the task is going to run
    void* data; // pointer to the data type the function takes
};

struct Task* create_task(void(*function)(void*), void* data);

#endif