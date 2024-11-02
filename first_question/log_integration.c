#include <stdio.h>
#include <omp.h>
#include <math.h>

// Function to calculate f(x) = ln(x) / x
double function(double x) {
    return log(x) / x;
}

int main() {
    double a = 1.0;  // Start of the interval
    double b = 10.0; // End of the interval
    int N;           // Variable to hold user input for N

    // Prompt user for input
    printf("Enter the number of rectangles (N): ");
    scanf("%d", &N);  // Read the value of N from user input

    double width = (b - a) / N;  // Width of each rectangle
    double total_area = 0.0;      // Variable to store the total area

    // Parallel region
    #pragma omp parallel
    {
        // Get the number of threads used
        int num_threads = omp_get_num_threads();
        
        // Print the number of threads (only in the master thread)
        #pragma omp master
        {
            printf("Number of threads used: %d\n", num_threads);
        }

        // Local variable for each thread
        double partial_area = 0.0;  

        // Parallel loop
        #pragma omp for
        for (int i = 0; i < N; i++) {
            double x = a + (i + 0.5) * width;  // Midpoint for better accuracy
            partial_area += function(x) * width; // Accumulate partial area for each thread
        }

        // Accumulate the partial areas into total_area
        #pragma omp atomic
        total_area += partial_area;
    }

    printf("For N = %d, the numerical integration of ln(x) / x over [1,10] is: %.10f\n", N, total_area);
    return 0;
}