#ifndef MATRIX_TEST_H
#define MATRIX_TEST_H

#include "../../src/nmath/nmatrix.h"
#include "../../libraries/logger/log.h"

void testMatrixCreation();
void test_get_sub_matrix_except_column();
void test_get_sub_matrix();
void test_serialize_matrix();

void split_matrix_test();
#endif