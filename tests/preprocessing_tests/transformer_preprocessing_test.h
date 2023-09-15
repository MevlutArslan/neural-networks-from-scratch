#ifndef TRANSFORMER_PREPROCESSING_TEST_H
#define TRANSFORMER_PREPROCESSING_TEST_H

#include "../../src/helper/data_processing.h"

void line_to_embedding_test();
void word_to_embedding_test();
void empty_word_to_embedding_test();
void fill_tokenizer_vocabulary_test();

void add_positional_embeddings_test();
void get_positional_embeddings_test();
/*

void add_positional_embeddings(MatrixArray* embeddings);
Matrix* get_positional_embeddings(Matrix* embedding, int d_model);
*/

#endif