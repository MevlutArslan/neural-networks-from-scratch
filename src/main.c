#include <stdio.h>
#include "nmath/nmath.h"
#include "helper/matrix_linked_list.h"
#include <string.h>
#include "../tests/test.h"

void runProgram();

int main(int argc, char* argv[])
{
    int isTesting = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "test") == 0) {
            isTesting = 1;
            break;
        }
    }

    if (isTesting) {
        printf("Running tests...\n");
        run_tests();
    } else {
        printf("Running the program...\n");
        runProgram();
    }
}

void runProgram() {
    MatrixLinkedListNode* matrixList = malloc(sizeof (MatrixLinkedListNode));
    Matrix* matrix = createMatrix(3,3);
    matrix->data[0][0] = 1;
    matrix->data[0][1] = 2;
    matrix->data[0][2] = 3;
    matrix->data[1][0] = 4;
    matrix->data[1][1] = 5;
    matrix->data[1][2] = 6;
    matrix->data[2][0] = 7;
    matrix->data[2][1] = 8;
    matrix->data[2][2] = 9;

    matrixList->next = NULL;
    matrixList->data = matrix;

    MatrixLinkedListNode* current = matrixList;
    while (current != NULL) {
        // printMatrix(current->data);
        freeMatrix(current->data);

        MatrixLinkedListNode* nextNode = current->next;
        free(current);
        current = nextNode;
    }
}