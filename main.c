#include <stdio.h>
#include "nmath/nmath.h"
#include "helper/matrix_linked_list.h"
#include "tests/helper_tests/linkedlist_test.h"
#include "tests/matrix_tests/matrix_test.h"
#include "tests/math_tests/matrix_operations.h"
#include <string.h>

void runTests();
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
        runTests();
    } else {
        printf("Running the program...\n");
        runProgram();
    }
    
}

void runTests() {
    testAddToTheEnd();
    testMatrixCreation();
    testMatrixMultiplication();
    testMatrixAddition();
    testMatrixSubtraction();
    testMatrixElementWiseMultiplication();
    testMatrixTranspose();
}

void runProgram() {
    MatrixLinkedListNode* matrixList = malloc(sizeof (MatrixLinkedListNode));
    Matrix* matrix = createMatrix(2,2);

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