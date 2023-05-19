#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void print_matrix(int rows, int cols, double **matrix)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
}

void matrix_multiply(int rows_a, int cols_a, double **a,
                     int rows_b, int cols_b, double **b,
                     double **c)
{
    for (int i = 0; i < rows_a; i++) {
        for (int j = 0; j < cols_b; j++) {
            double sum = 0.0;
            for (int k = 0; k < cols_a; k++) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
}

int main()
{
    const int ROWS_A = 1000;
    const int COLS_A = 1000;
    const int ROWS_B = 1000;
    const int COLS_B = 1000;

    double **a = (double **)malloc(ROWS_A * sizeof(double *));
    double **b = (double **)malloc(ROWS_B * sizeof(double *));
    double **c = (double **)malloc(ROWS_A * sizeof(double *));
    for (int i = 0; i < ROWS_A; i++) {
        a[i] = (double *)malloc(COLS_A * sizeof(double));
        b[i] = (double *)malloc(COLS_B * sizeof(double));
        c[i] = (double *)malloc(COLS_B * sizeof(double));
    }

    // Initialize matrices with random values
    srand(time(NULL));
    for (int i = 0; i < ROWS_A; i++) {
        for (int j = 0; j < COLS_A; j++) {
            a[i][j] = rand() % 10;
        }
    }

    for (int i = 0; i < ROWS_B; i++) {
        for (int j = 0; j < COLS_B; j++) {
            b[i][j] = rand() % 10;
        }
    }

    clock_t start = clock();
    matrix_multiply(ROWS_A, COLS_A, a, ROWS_B, COLS_B, b, c);
    clock_t end = clock();

    double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken: %f seconds\n", elapsed_time);

    // Cleanup: free allocated memory
    for (int i = 0; i < ROWS_A; i++) {
        free(a[i]);
        free(b[i]);
        free(c[i]);
    }
    free(a);
    free(b);
    free(c);

    return 0;
}
