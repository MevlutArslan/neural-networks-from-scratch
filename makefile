CC = gcc
CFLAGS = -Wall -Wextra -g -DWITH_X11 
LDFLAGS = -lm

BUILD_DIR = build
SRC_DIR = src
LIBRARY_DIR = libraries
TEST_DIR = tests

SRC_FILES = $(wildcard $(SRC_DIR)/*.c) $(wildcard $(SRC_DIR)/helper/*.c) $(wildcard $(SRC_DIR)/helper/data_processing/*.c) $(wildcard $(SRC_DIR)/nmath/*.c) $(wildcard $(SRC_DIR)/neural_network/*.c) $(wildcard $(SRC_DIR)/neural_network/loss_functions/*.c)  $(wildcard $(SRC_DIR)/neural_network/loss_functions/*.c) $(wildcard $(SRC_DIR)/example_networks/*.c) $(wildcard $(SRC_DIR)/example_networks/mnist/*.c) $(wildcard $(SRC_DIR)/example_networks/wine_dataset/*.c)
# SRC_FILES := $(shell find $(SRC_DIR)/example_networks -name '*.c')

LIBRARY_FILES = $(wildcard $(LIBRARY_DIR)/gnuplot_i/*.c) $(wildcard $(LIBRARY_DIR)/logger/*.c) 
TEST_FILES = $(wildcard $(TEST_DIR)/*.c) $(wildcard $(TEST_DIR)/math_tests/*.c) $(wildcard $(TEST_DIR)/matrix_tests/*.c) $(wildcard $(TEST_DIR)/neural_network_tests/*.c)

OBJ_FILES = $(addprefix $(BUILD_DIR)/, $(notdir $(SRC_FILES:.c=.o)))
OBJ_FILES += $(addprefix $(BUILD_DIR)/, $(notdir $(LIBRARY_FILES:.c=.o)))
OBJ_FILES += $(addprefix $(BUILD_DIR)/, $(notdir $(TEST_FILES:.c=.o)))

TARGET = $(BUILD_DIR)/program

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ_FILES)
	$(CC) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/helper/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/helper/data_processing/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/neural_network/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/neural_network/activation_functions/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/neural_network/loss_functions/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/example_networks/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/example_networks/mnist/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/example_networks/wine_dataset/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/nmath/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(LIBRARY_DIR)/gnuplot_i/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DWITH_X11 -I/opt/X11/include -L/opt/X11/lib -c $< -o $@

$(BUILD_DIR)/%.o: $(LIBRARY_DIR)/logger/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/math_tests/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/matrix_tests/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/neural_network_tests/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)/*.o $(TARGET)
