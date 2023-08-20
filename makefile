CC = gcc
NVCC = nvcc

CFLAGS = -g
NVCCFLAGS = -G -g
# tests/*.c tests/**/*.c
C_FILES = $(wildcard src/*.c src/example_networks/**/*.c src/**/*.c libraries/**/*.c )
CU_FILES = $(wildcard src/nmath/*.cu)

OBJ_FILES = $(addprefix build/, $(addsuffix .o, $(basename $(C_FILES) $(CU_FILES))))

main: $(OBJ_FILES)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

build/%.o: %.c
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

build/%.o: %.cu
	mkdir -p $(@D)
	$(NVCC) -c $< -o $@

clean:
	rm -rf build main