#ifndef DATA_PROCESSING_H
#define DATA_PROCESSING_H

#include <csv.h>
#include "../nmath/nmath.h"
#include "../helper/hashmap.h"
#include <stdlib.h>
// TODO: Move data separation out of this
typedef struct {
    int rows;
    int columns;
    Matrix* data;
} Data;

typedef enum NormalizationMethod {
    STANDARD_DEVIATION, BY_DIVISION
} NormalizationMethod;

Data* load_csv(char* file_location);

MatrixArray* load_text_as_embedding(char* file_location, map_t char_int_map, map_t int_char_map, int max_sequence_length, int vocab_size);
void fill_tokenizer_vocabulary(char* text, map_t char_int_map, map_t int_char_map, int* index);
Matrix* line_to_embedding(char* text, int max_sequence_length, int vocab_size, map_t char_int_map);
void word_to_embedding(char* token, map_t char_int_map, Vector* vector);

void add_positional_embeddings(MatrixArray* embeddings);
Matrix* get_positional_embeddings(Matrix* embedding, int num_rows, int d_model);

void removeResultsFromEvaluationSet(Matrix* eval_matrix, int column_index);
Vector* extractYValues(Matrix* training_matrix, int column_index);
Matrix* oneHotEncode(Vector* categories, int num_categories);

int getRowCount(char* file_location);
int getColumnCount(char* file_location);

void normalize_column(NormalizationMethod normalization_method, Matrix* matrix, int column_index, float division_factor);
void normalize_column_standard_deviation(Matrix* matrix, int column_index);
void normalize_column_by_division(Matrix* matrix, int column_index, float division_factor);

void normalizeVector(Vector* vector);

void unnormalizeColumn(Matrix* vector, int column_index, double min, double max);

void unnormalizeVector(Vector* vector, double min, double max);

void free_data(Data* data);

#endif