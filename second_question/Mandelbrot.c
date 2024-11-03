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
        
        double x = (double)rand() / RAND_MAX * (R_MAX - R_MIN) + R_MIN;
        double y = (double)rand() / RAND_MAX * (I_MAX - I_MIN) + I_MIN;
        if (is_in_mandelbrot(x, y)) count++;
    }
    // Return the area approximation
    return (R_MAX - R_MIN) * (I_MAX - I_MIN) * ((double)count / num_points);
}

// Function for the parallel version of Monte Carlo Mandelbrot area calculation using OpenMP
double mandelbrot_parallel(int num_points, int num_threads) {
    int count = 0;
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        int local_count = 0;
        unsigned int seed = omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < num_points; i++) {
            double x = (double)rand_r(&seed) / RAND_MAX * (R_MAX - R_MIN) + R_MIN;
            double y = (double)rand_r(&seed) / RAND_MAX * (I_MAX - I_MIN) + I_MIN;
            if (is_in_mandelbrot(x, y)) local_count++;
        }
        #pragma omp atomic
        count += local_count;
    }
    return (R_MAX - R_MIN) * (I_MAX - I_MIN) * ((double)count / num_points);
}

int main() {
    int num_points = NUM_POINTS;
    int thread_counts[] = {2, 4, 8, 16, 32};  
    int num_threads_count = sizeof(thread_counts) / sizeof(thread_counts[0]);
    double start, end;

    // Measure and print the execution time for the serial version
    start = omp_get_wtime();
    double area_serial = mandelbrot_serial(num_points);
    end = omp_get_wtime();
    printf("Serial Mandelbrot area: %f\n", area_serial);
    printf("Time (serial): %f seconds\n", end - start);

    // Loop through different thread counts and measure execution time
    for (int i = 0; i < num_threads_count; i++) {
        int num_threads = thread_counts[i];
        
        start = omp_get_wtime();
        double area_parallel = mandelbrot_parallel(num_points, num_threads);
        end = omp_get_wtime();
        
        printf("Parallel Mandelbrot area with %d threads: %f\n", num_threads, area_parallel);
        printf("Time (parallel with %d threads): %f seconds\n", num_threads, end - start);
    }

    return 0;
}
