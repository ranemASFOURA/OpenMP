# Makefile for building, running, and profiling the Mandelbrot area calculation

CC = gcc
CFLAGS = -fopenmp -pg -O2  
TARGET = Mandelbrot

all: $(TARGET)

$(TARGET): Mandelbrot.c
	$(CC) $(CFLAGS) -o $(TARGET) Mandelbrot.c

# Create a 'run' target to execute the program and profile
run: $(TARGET)
	@echo "Running the program..."
	./$(TARGET)
	@echo "Profiling with gprof..."
	gprof $(TARGET) gmon.out > profile_report.txt
	@echo "Profiling complete. See profile_report.txt for details."

# Create a 'profile' target that runs the program and generates the profile
profile: run

clean:
	rm -f $(TARGET) gmon.out profile_report.txt
