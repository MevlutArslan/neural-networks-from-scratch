#include "thread_pool.h"
#include <bits/pthreadtypes.h>
#include <pthread.h>

void push_task(struct ThreadPool* thread_pool, struct Task* task);
struct Task* pop_task(struct ThreadPool* thread_pool);

struct ThreadPool* create_thread_pool(int num_threads) {
    struct ThreadPool* thread_pool = (struct ThreadPool*) malloc(sizeof(struct ThreadPool));
    thread_pool->is_active = 1;
    thread_pool->num_threads = num_threads;
    thread_pool->threads = (pthread_t*) malloc(num_threads * sizeof(pthread_t));
    thread_pool->num_pending_tasks = 0;
    thread_pool->num_threads_exited = 0;

    thread_pool->task_queue = init_task_queue(200);

    thread_pool->lock = (pthread_mutex_t) PTHREAD_MUTEX_INITIALIZER;
    thread_pool->pending_tasks_lock = (pthread_mutex_t) PTHREAD_MUTEX_INITIALIZER;
    thread_pool->capacity_lock = (pthread_mutex_t) PTHREAD_MUTEX_INITIALIZER;

    thread_pool->capacity_signal = (pthread_cond_t) PTHREAD_COND_INITIALIZER;
    thread_pool->task_available_cond = (pthread_cond_t) PTHREAD_COND_INITIALIZER;
    thread_pool->all_tasks_done = (pthread_cond_t) PTHREAD_COND_INITIALIZER;

    thread_pool->pop_task = pop_task;
    thread_pool->push_task = push_task;

    for(int i = 0; i < num_threads; i++) {
        pthread_create(&thread_pool->threads[i], NULL, run_thread, thread_pool);
    }
    
    return thread_pool;
}

void* run_thread(void* args) {
    struct ThreadPool* thread_pool = (struct ThreadPool*) args;
    struct Task* task = NULL;

    while(thread_pool->is_active == 1) {    
        // Lock to coordinate thread states: waiting for a task or dequeuing a task from the queue
        pthread_mutex_lock(&thread_pool->lock);
        while(thread_pool->task_queue->element_count == 0 && thread_pool->is_active == 1){
            pthread_cond_wait(&thread_pool->task_available_cond, &thread_pool->lock);
            // log_info("thread: %lu is waiting for tasks", pthread_self());
        }
        if(thread_pool->is_active == 0) {
            pthread_mutex_unlock(&thread_pool->lock);
            break;
        }
        task = thread_pool->pop_task(thread_pool);
        thread_pool->task_queue->element_count--;
        // log_info("thread: %lu is got a task", pthread_self());

        pthread_mutex_unlock(&thread_pool->lock);

        task->function(task->data);

        // lock that checks if we are done with the work to signal all_tasks_done cond.
        pthread_mutex_lock(&thread_pool->pending_tasks_lock);

        thread_pool->num_pending_tasks --;
        if(thread_pool->num_pending_tasks <= 0) {
            pthread_cond_signal(&thread_pool->all_tasks_done);
            // log_info("thread: %lu signalled all tasks are done!", pthread_self());
        }
        pthread_mutex_unlock(&thread_pool->pending_tasks_lock);
    }

    return NULL;
}

void wait_for_all_tasks(struct ThreadPool* thread_pool) {
    pthread_mutex_lock(&thread_pool->pending_tasks_lock);
    while(thread_pool->num_pending_tasks > 0) {
        pthread_cond_wait(&thread_pool->all_tasks_done, &thread_pool->pending_tasks_lock);
    }

    // log_info("thread_pool is done with tasks! returning.");
    pthread_mutex_unlock(&thread_pool->pending_tasks_lock);
}

struct TaskQueue* init_task_queue(int capacity) {
    struct TaskQueue* task_queue = (struct TaskQueue*) calloc(1, sizeof(struct TaskQueue));
    
    task_queue->tasks = (struct Task**) malloc(capacity * sizeof(struct Task*));
    task_queue->capacity = capacity;
    task_queue->element_count = 0;

    return task_queue;
}

void push_task(struct ThreadPool* thread_pool, struct Task* task) {
    pthread_mutex_lock(&thread_pool->lock);
    while(thread_pool->task_queue->element_count >= thread_pool->task_queue->capacity) {
        printf("At full capacity, waiting for others tasks to finish...");
        pthread_cond_wait(&thread_pool->capacity_signal, &thread_pool->lock);
    }

    thread_pool->task_queue->tasks[thread_pool->task_queue->element_count] = task;
    thread_pool->task_queue->element_count ++;

    pthread_mutex_lock(&thread_pool->pending_tasks_lock);

    thread_pool->num_pending_tasks++;

    pthread_mutex_unlock(&thread_pool->pending_tasks_lock);

    pthread_cond_broadcast(&thread_pool->task_available_cond);

    pthread_mutex_unlock(&thread_pool->lock);
}


struct Task* pop_task(struct ThreadPool* thread_pool) {
    struct Task* task = thread_pool->task_queue->tasks[0];

    for(int i = 0; i < thread_pool->task_queue->element_count - 1; i++) {
        thread_pool->task_queue->tasks[i] = thread_pool->task_queue->tasks[i + 1];
    }
    thread_pool->task_queue->tasks[thread_pool->task_queue->element_count - 1] = NULL;
    
    pthread_cond_signal(&thread_pool->capacity_signal);  // signal that a task has been popped from the queue

    return task;
}


void destroy_thread_pool(struct ThreadPool* thread_pool) {
    thread_pool->is_active = 0;

    // wake up each thread from their sleep caused by lack of tasks and trigger exit from the while loop
    pthread_cond_broadcast(&thread_pool->task_available_cond);
    
    for(int i = 0; i < thread_pool->num_threads; i++) {
        pthread_join(thread_pool->threads[i], NULL);
    }
    for(int i = 0; i < thread_pool->task_queue->element_count; i++) {
        free(thread_pool->task_queue->tasks[i]);
    }

    free(thread_pool->threads);

    free(thread_pool->task_queue->tasks);
    free(thread_pool->task_queue);

    pthread_mutex_destroy(&thread_pool->lock);
    pthread_mutex_destroy(&thread_pool->capacity_lock);
    pthread_mutex_destroy(&thread_pool->pending_tasks_lock);

    pthread_cond_destroy(&thread_pool->capacity_signal);
    pthread_cond_destroy(&thread_pool->all_tasks_done);
    pthread_cond_destroy(&thread_pool->task_available_cond);

    free(thread_pool);
}


struct Task* create_task(void(*function)(void*), void* data) {
    struct Task* task = (struct Task*) malloc(sizeof(struct Task));

    task->function = function;
    task->data = data;

    return task;
}