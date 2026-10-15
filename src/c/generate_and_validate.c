#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static uint64_t lcg_state = 42;

void lcg_seed(uint64_t seed) {
    lcg_state = seed & 0xFFFFFFFFULL;
}

double lcg_next_double(void) {
    lcg_state = (1664525ULL * lcg_state + 1013904223ULL) & 0xFFFFFFFFULL;
    return ((double)lcg_state / 4294967296.0) * 2.0 - 1.0;
}

void matmul_naive(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void write_test_case(FILE *fp, const char *name, int n, const double *A, const double *B, const double *C) {
    fprintf(fp, "TEST %s %d\n", name, n);
    for (int i = 0; i < n * n; i++) fprintf(fp, "%.17g%c", A[i], (i + 1 == n * n) ? '\n' : ' ');
    for (int i = 0; i < n * n; i++) fprintf(fp, "%.17g%c", B[i], (i + 1 == n * n) ? '\n' : ' ');
    for (int i = 0; i < n * n; i++) fprintf(fp, "%.17g%c", C[i], (i + 1 == n * n) ? '\n' : ' ');
}

int main(const int argc, const char **argv) {
    const char *out_path = (argc > 1) ? argv[1] : "data/validation_suite.txt";
    FILE *fp = fopen(out_path, "w");
    if (!fp) {
        perror("Failed to open validation output file");
        return EXIT_FAILURE;
    }

    fprintf(fp, "4\n");

    {
        int n = 4;
        double A[16] = {
            2.5, -1.0,  0.5,  4.0,
            0.0,  3.2, -2.1,  1.1,
           -4.5,  0.8,  1.5, -0.5,
            1.0,  2.0,  3.0,  4.0
        };
        double B[16] = {0};
        double C[16] = {0};
        for (int i = 0; i < n; i++) B[i * n + i] = 1.0;
        matmul_naive(A, B, C, n);
        write_test_case(fp, "IDENTITY_4x4", n, A, B, C);
    }

    {
        int n = 4;
        double A[16] = {
            1.5, -2.3,  4.1,  0.9,
           -3.2,  5.0, -1.1,  2.2,
            0.4,  0.8, -6.5,  3.3,
            7.1, -0.2,  1.9, -4.4
        };
        double B[16] = {0};
        double C[16] = {0};
        matmul_naive(A, B, C, n);
        write_test_case(fp, "ZERO_4x4", n, A, B, C);
    }

    {
        int n = 3;
        double A[9] = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        };
        double B[9] = {
            9.0, 8.0, 7.0,
            6.0, 5.0, 4.0,
            3.0, 2.0, 1.0
        };
        double C[9] = {0};
        matmul_naive(A, B, C, n);
        write_test_case(fp, "KNOWN_3x3", n, A, B, C);
    }

    {
        int n = 32;
        double *A = malloc(n * n * sizeof(double));
        double *B = malloc(n * n * sizeof(double));
        double *C = malloc(n * n * sizeof(double));
        lcg_seed(42);
        for (int i = 0; i < n * n; i++) A[i] = lcg_next_double();
        for (int i = 0; i < n * n; i++) B[i] = lcg_next_double();
        matmul_naive(A, B, C, n);
        write_test_case(fp, "LCG_SEED42_32x32", n, A, B, C);
        free(A);
        free(B);
        free(C);
    }

    fclose(fp);
    printf("[C Generator] Successfully wrote 4 validation test cases to %s\n", out_path);
    return EXIT_SUCCESS;
}
