CC = gcc
CFLAGS = -Wall -Wextra

BUILD_DIR = build
SRC_DIR = src
TEST_DIR = tests

SRC_FILES = $(wildcard $(SRC_DIR)/*.c) $(wildcard $(SRC_DIR)/helper/*.c) $(wildcard $(SRC_DIR)/nmath/*.c) $(wildcard $(SRC_DIR)/neural_network/*.c)
TEST_FILES = $(wildcard $(TEST_DIR)/*.c) $(wildcard $(TEST_DIR)/helper_tests/*.c) $(wildcard $(TEST_DIR)/math_tests/*.c) $(wildcard $(TEST_DIR)/matrix_tests/*.c)

OBJ_FILES = $(addprefix $(BUILD_DIR)/, $(notdir $(SRC_FILES:.c=.o)))
OBJ_FILES += $(addprefix $(BUILD_DIR)/, $(notdir $(TEST_FILES:.c=.o)))

TARGET = $(BUILD_DIR)/program

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ_FILES)
	$(CC) $^ -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/helper/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/neural_network/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/nmath/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/helper_tests/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/math_tests/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/matrix_tests/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)/*.o $(TARGET)
