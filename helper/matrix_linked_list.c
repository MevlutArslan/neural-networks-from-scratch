#include "matrix_linked_list.h"

void addToTheEnd(MatrixLinkedListNode* node, Matrix* m){
    MatrixLinkedListNode* current = node;

    while(current->next != NULL) {
        current = current->next;
    }

    MatrixLinkedListNode* newNode = malloc(sizeof(MatrixLinkedListNode));
    newNode->next = NULL;
    newNode->data = m;

    current->next = newNode;
}