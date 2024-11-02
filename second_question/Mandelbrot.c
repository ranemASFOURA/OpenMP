#include <stdio.h>
#include <stdlib.h>
#include <omp.h>  

// Define the number of random points for the Monte Carlo method
#define NUM_POINTS 10000000

// Define the ranges for the Mandelbrot set
#define R_MAX 2.0
#define R_MIN -2.0
#define I_MAX 1.0
#define I_MIN -1.0

// Function to check if the point (x, y) belongs to the Mandelbrot set
int is_in_mandelbrot(double x, double y) {
    double real = x;
    double imag = y;
    // Iterate to check if the point escapes
    for (int i = 0; i < 1000; i++) {
        double temp_real = real * real - imag * imag + x;
        imag = 2 * real * imag + y;
        real = temp_real;
        if (real * real + imag * imag > 4.0) return 0;  // Escapes the set
    }
    return 1;  // Remains within the set
}

// Function for the sequential version of Monte Carlo Mandelbrot area calculation
double mandelbrot_serial(int num_points) {
    int count = 0;  // Counter for points inside the Mandelbrot set
    for (int i = 0; i < num_points; i++) {
        // Generate random points within the defined ranges
        double x = (double)rand() / RAND_MAX * (R_MAX - R_MIN) + R_MIN;
        double y = (double)rand() / RAND_MAX * (I_MAX - I_MIN) + I_MIN;
        if (is_in_mandelbrot(x, y)) count++;
    }
    // Return the area approximation
    return (R_MAX - R_MIN) * (I_MAX - I_MIN) * ((double)count / num_points);
}

// Function for the parallel version of Monte Carlo Mandelbrot area calculation using OpenMP
double mandelbrot_parallel(int num_points, int num_threads) {
    int count = 0;  // Counter for points inside the Mandelbrot set
    omp_set_num_threads(num_threads);  // Set the number of threads
    #pragma omp parallel  // Start a parallel region
    {
        int local_count = 0;  // Local counter for each thread
        #pragma omp for  // Parallelize the loop
        for (int i = 0; i < num_points; i++) {
            // Generate random points within the defined ranges
            double x = (double)rand() / RAND_MAX * (R_MAX - R_MIN) + R_MIN;
            double y = (double)rand() / RAND_MAX * (I_MAX - I_MIN) + I_MIN;
            if (is_in_mandelbrot(x, y)) local_count++;
        }
        #pragma omp atomic  // Ensure safe update of the global counter
        count += local_count;
    }
    // Return the area approximation
    return (R_MAX - R_MIN) * (I_MAX - I_MIN) * ((double)count / num_points);
}

int main() {
    // Variables for measuring execution time
    double start, end;
    int num_threads;

    // Prompt the user to enter the number of threads
    printf("Enter the number of threads: ");
    if (scanf("%d", &num_threads) != 1) {
        fprintf(stderr, "Error reading the number of threads.\n");
        return 1;  // Exit with an error code
    }

    // Measure and print the execution time for the serial version
    start = omp_get_wtime();
    double area_serial = mandelbrot_serial(NUM_POINTS);
    end = omp_get_wtime();
    printf("Serial Mandelbrot area: %f\n", area_serial);
    printf("Time (serial): %f seconds\n", end - start);

    // Measure and print the execution time for the parallel version
    start = omp_get_wtime();
    double area_parallel = mandelbrot_parallel(NUM_POINTS, num_threads);
    end = omp_get_wtime();
    printf("Parallel Mandelbrot area: %f\n", area_parallel);
    printf("Time (parallel with %d threads): %f seconds\n", num_threads, end - start);

    return 0;
}
