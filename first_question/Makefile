# Compiler and flags
CC = gcc
CFLAGS = -fopenmp -pg -Wall -lm

# Executable and source files
EXEC = log_integration
SRC = log_integration.c

# Target to build the program
all: $(EXEC)

$(EXEC): $(SRC)
	$(CC) $(SRC) $(CFLAGS) -o $(EXEC)

# Target to run the program
run: $(EXEC)
	./$(EXEC)

# Target to profile the program using gprof
profile: $(EXEC)
	./$(EXEC)
	gprof $(EXEC) gmon.out > profile.txt

# Target to clean the generated files
clean:
	rm -f $(EXEC) gmon.out profile.txt