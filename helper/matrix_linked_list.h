#ifndef LINKED_LIST_H
#define LINKED_LIST_H

#include "../nmath/nmath.h"

typedef struct MatrixLinkedListNode{
    struct MatrixLinkedListNode* next;
    Matrix* data;
} MatrixLinkedListNode;

void addToTheEnd(MatrixLinkedListNode* node, Matrix* m);

#endif