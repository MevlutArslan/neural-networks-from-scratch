CC = gcc

CFLAGS = -g -Wall

C_FILES = $(wildcard src/*.c src/networks/**/*.c src/**/*.c libraries/**/*.c tests/*.c tests/**/*.c)

OBJ_FILES = $(addprefix build/, $(addsuffix .o, $(basename $(C_FILES))))

main: $(OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ $^

build/%.o: %.c
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf build main
