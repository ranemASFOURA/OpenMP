#include <stdio.h>
#include <omp.h>
#include <math.h>

double function(double x) {
    return log(x) / x;
}

// Function to perform numerical integration
double integrate(int N, double a, double b, int num_threads) {
    double width = (b - a) / N;       // Width of each rectangle
    double total_area = 0.0;          // Variable to store the total area

   
    omp_set_num_threads(num_threads);

    // Parallel region with reduction
    #pragma omp parallel reduction(+:total_area)
    {
        // Local variable for each thread
        double partial_area = 0.0;

        // Parallel loop with dynamic schedule
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < N; i++) {
            double x = a + (i + 0.5) * width;  // Midpoint for better accuracy
            partial_area += function(x) * width; 
        }

       
        total_area += partial_area; // This will be handled by the reduction
    }

    return total_area;
}

int main() {
    double a = 1.0;  // Start of the interval
    double b = 10.0; // End of the interval
    int N;           // Variable to hold user input for N

    
    printf("Enter the number of rectangles (N): ");
    scanf("%d", &N);  

    // Loop over different thread counts
    for (int num_threads = 2; num_threads <= 8; num_threads *= 2) {
        double total_area = integrate(N, a, b, num_threads);
        printf("For N = %d and using %d threads, the numerical integration of ln(x) / x over [1,10] is: %.10f\n", N, num_threads, total_area);
    }

    return 0;
}