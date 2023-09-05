#ifndef OPTIMIZATION_ALGORITHM_TEST_H
#define OPTIMIZATION_ALGORITHM_TEST_H

#include "../../src/neural_network/nnetwork.h"
#include <assert.h>

NNetwork* create_mock_network(OptimizationConfig* optimization_config);

void test_mock_sgd();
void test_mock_sgd_momentum();
void test_mock_adagrad();
void test_mock_rms_prop();
void test_mock_adam();

#endif