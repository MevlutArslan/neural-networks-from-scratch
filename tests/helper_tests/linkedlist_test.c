#include "linkedlist_test.h"

void testAddToTheEnd() {
    // Create the initial linked list node with a matrix
    Matrix* matrix1 = createMatrix(1, 1);

    MatrixLinkedListNode* head = malloc(sizeof(MatrixLinkedListNode));
    head->next = NULL;
    head->data = matrix1;

    // Add more matrices to the end of the linked list
    Matrix* matrix2 = createMatrix(2, 2);
    Matrix* matrix3 = createMatrix(3, 4);

    addToTheEnd(head, matrix2);
    addToTheEnd(head, matrix3);

    // Traverse the linked list and print the matrices
    MatrixLinkedListNode* current = head;
    int counter = 1;
    while (current != NULL) {
        if(counter != current->data->rows) {
           printf("Failed while testing addToTheEnd() function! \n");
        }
        current = current->next;
        counter++;
    }

    // Free the memory
    current = head;
    while (current != NULL) {
        MatrixLinkedListNode* nextNode = current->next;
        freeMatrix(current->data);
        free(current);
        current = nextNode;
    }

    printf("Successfully tested the addToTheEnd() function! \n");

}