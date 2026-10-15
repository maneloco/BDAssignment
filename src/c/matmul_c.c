// src/c/matmul_c.c
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

static uint64_t lcg_state = 42;

void lcg_seed(uint64_t seed) {
    lcg_state = seed & 0xFFFFFFFFULL;
}

double lcg_next_double(void) {
    lcg_state = (1664525ULL * lcg_state + 1013904223ULL) & 0xFFFFFFFFULL;
    return ((double)lcg_state / 4294967296.0) * 2.0 - 1.0;
}

// Pure unoptimized triple-loop O(n^3) kernel (i-j-k order, 1D row-major layout)
void matmul_kernel(const double *A, const double *B, double *C, int n) {
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

long get_peak_rss_kb(void) {
    FILE *fp = fopen("/proc/self/status", "r");
    if (!fp) return -1;
    char line[256];
    long vmrss = -1;
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "VmHWM:", 6) == 0) {
            sscanf(line + 6, "%ld", &vmrss);
            break;
        }
    }
    fclose(fp);
    return vmrss;
}

int validate_from_file(const char *filepath) {
    FILE *fp = fopen(filepath, "r");
    if (!fp) {
        perror("Cannot open validation file");
        return 0;
    }

    int num_tests = 0;
    if (fscanf(fp, "%d", &num_tests) != 1) {
        fclose(fp);
        return 0;
    }

    const double atol = 1e-9;
    const double rtol = 1e-9;

    for (int t = 0; t < num_tests; t++) {
        char tag[32], name[64];
        int n = 0;
        if (fscanf(fp, "%31s %63s %d", tag, name, &n) != 3) {
            fclose(fp);
            return 0;
        }

        double *A = malloc(n * n * sizeof(double));
        double *B = malloc(n * n * sizeof(double));
        double *C_ref = malloc(n * n * sizeof(double));
        double *C_calc = calloc(n * n, sizeof(double));

        for (int i = 0; i < n * n; i++) fscanf(fp, "%lf", &A[i]);
        for (int i = 0; i < n * n; i++) fscanf(fp, "%lf", &B[i]);
        for (int i = 0; i < n * n; i++) fscanf(fp, "%lf", &C_ref[i]);

        if (strcmp(name, "LCG_SEED42_32x32") == 0) {
            lcg_seed(42);
            for (int i = 0; i < n * n; i++) {
                if (fabs(A[i] - lcg_next_double()) > 1e-15) return 0;
            }
            for (int i = 0; i < n * n; i++) {
                if (fabs(B[i] - lcg_next_double()) > 1e-15) return 0;
            }
        }

        matmul_kernel(A, B, C_calc, n);

        double max_diff = 0.0;
        for (int i = 0; i < n * n; i++) {
            double diff = fabs(C_calc[i] - C_ref[i]);
            double tol = atol + rtol * fabs(C_ref[i]);
            if (diff > max_diff) max_diff = diff;
            if (diff > tol) {
                fprintf(stderr, "[C Validation FAIL] %s index %d\n", name, i);
                return 0;
            }
        }
        printf("[C Validation PASS] %-18s (n=%2d) | max_abs_error = %.3e\n", name, n, max_diff);

        free(A); free(B); free(C_ref); free(C_calc);
    }

    fclose(fp);
    return 1;
}

int main(int argc, char **argv) {
    const char *val_file = "data/validation_suite.txt";
    const char *csv_file = "results/raw_measurements.csv";
    const char *opt_label = (argc > 1) ? argv[1] : "C_O2_row_major_1D";

    if (!validate_from_file(val_file)) return EXIT_FAILURE;

    FILE *csv = fopen(csv_file, "a");
    if (!csv) return EXIT_FAILURE;

    int sizes[] = {64, 128, 256, 512, 768, 1024, 1536, 2048};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int warmup_runs = 2;
    int measured_runs = 5;
    int threads = 1;
    double max_budget_sec = 300.0;

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        size_t num_elements = (size_t)n * (size_t)n;
        double *A = malloc(num_elements * sizeof(double));
        double *B = malloc(num_elements * sizeof(double));
        double *C = malloc(num_elements * sizeof(double));

        if (!A || !B || !C) {
            fprintf(csv, "C,%s,%d,float64,%d,%d,0,NaN,%.1f,%ld,NaN,NaN,MEMORY_FAILURE\n",
                    opt_label, n, threads, warmup_runs, max_budget_sec, get_peak_rss_kb());
            free(A); free(B); free(C);
            break;
        }

        lcg_seed(42);
        for (size_t i = 0; i < num_elements; i++) A[i] = lcg_next_double();
        for (size_t i = 0; i < num_elements; i++) B[i] = lcg_next_double();

        for (int w = 0; w < warmup_runs; w++) {
            memset(C, 0, num_elements * sizeof(double));
            matmul_kernel(A, B, C, n);
        }

        int timed_out = 0;
        for (int rep = 1; rep <= measured_runs; rep++) {
            memset(C, 0, num_elements * sizeof(double));

            struct timespec t_start, t_end;
            clock_gettime(CLOCK_MONOTONIC, &t_start);
            matmul_kernel(A, B, C, n);
            clock_gettime(CLOCK_MONOTONIC, &t_end);

            double elapsed = (t_end.tv_sec - t_start.tv_sec) +
                             (t_end.tv_nsec - t_start.tv_nsec) * 1e-9;

            double checksum = 0.0;
            for (size_t i = 0; i < num_elements; i++) checksum += C[i];

            long peak_rss = get_peak_rss_kb();
            double array_mem_kb = (3.0 * num_elements * sizeof(double)) / 1024.0;
            const char *status = (elapsed <= max_budget_sec) ? "OK" : "TIMEOUT_BUDGET_EXCEEDED";

            fprintf(csv, "C,%s,%d,float64,%d,%d,%d,%.9f,%.1f,%ld,%.2f,%.6f,%s\n",
                    opt_label, n, threads, warmup_runs, rep, elapsed, max_budget_sec,
                    peak_rss, array_mem_kb, checksum, status);
            fflush(csv);

            printf("[C Benchmark] n=%4d | rep=%d/%d | time=%.6f s | PeakRSS=%ld kB | checksum=%.4f\n",
                   n, rep, measured_runs, elapsed, peak_rss, checksum);

            if (elapsed > max_budget_sec) {
                timed_out = 1;
                break;
            }
        }

        free(A); free(B); free(C);
        if (timed_out) break;
    }

    fclose(csv);
    return EXIT_SUCCESS;
}
