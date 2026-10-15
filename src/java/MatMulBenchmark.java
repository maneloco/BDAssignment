// src/java/MatMulBenchmark.java
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Locale;
import java.util.Scanner;

public class MatMulBenchmark {

    private static long lcgState = 42L;

    public static void lcgSeed(long seed) {
        lcgState = seed & 0xFFFFFFFFL;
    }

    public static double lcgNextDouble() {
        lcgState = (1664525L * lcgState + 1013904223L) & 0xFFFFFFFFL;
        return ((double) lcgState / 4294967296.0) * 2.0 - 1.0;
    }

    // Pure unoptimized triple-loop O(n^3) kernel (i-j-k order)
    public static void matmulKernel(double[][] A, double[][] B, double[][] C, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
    }

    public static long getPeakRssKb() {
        try (BufferedReader br = new BufferedReader(new FileReader("/proc/self/status"))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.startsWith("VmHWM:")) {
                    String[] parts = line.trim().split("\\s+");
                    return Long.parseLong(parts[1]);
                }
            }
        } catch (Exception e) {
            return -1L;
        }
        return -1L;
    }

    public static boolean validateFromFile(String filepath) {
        final double atol = 1e-9;
        final double rtol = 1e-9;

        try (Scanner scanner = new Scanner(new BufferedReader(new FileReader(filepath)))) {
            scanner.useLocale(Locale.US);
            if (!scanner.hasNextInt()) return false;
            int numTests = scanner.nextInt();

            for (int t = 0; t < numTests; t++) {
                String tag = scanner.next();
                String name = scanner.next();
                int n = scanner.nextInt();

                double[][] A = new double[n][n];
                double[][] B = new double[n][n];
                double[][] CRef = new double[n][n];
                double[][] CCalc = new double[n][n];

                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        A[i][j] = scanner.nextDouble();

                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        B[i][j] = scanner.nextDouble();

                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        CRef[i][j] = scanner.nextDouble();

                if ("LCG_SEED42_32x32".equals(name)) {
                    lcgSeed(42L);
                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < n; j++)
                            if (Math.abs(A[i][j] - lcgNextDouble()) > 1e-15) return false;
                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < n; j++)
                            if (Math.abs(B[i][j] - lcgNextDouble()) > 1e-15) return false;
                }

                matmulKernel(A, B, CCalc, n);

                double maxDiff = 0.0;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double diff = Math.abs(CCalc[i][j] - CRef[i][j]);
                        double tol = atol + rtol * Math.abs(CRef[i][j]);
                        if (diff > maxDiff) maxDiff = diff;
                        if (diff > tol) return false;
                    }
                }
                System.out.printf(Locale.US, "[Java Validation PASS] %-18s (n=%2d) | max_abs_error = %.3e%n",
                        name, n, maxDiff);
            }
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
        return true;
    }

    public static void main(String[] args) {
        Locale.setDefault(Locale.US);
        String valFile = "data/validation_suite.txt";
        String csvFile = "results/raw_measurements.csv";
        String methodLabel = "Java_2D_array_ijk";

        if (!validateFromFile(valFile)) {
            System.exit(1);
        }

        int[] sizes = {64, 128, 256, 512, 768, 1024, 1536, 2048};
        int warmupRuns = 2;
        int measuredRuns = 5;
        int threads = 1;
        double maxBudgetSec = 300.0;

        try (PrintWriter csv = new PrintWriter(new FileWriter(csvFile, true))) {
            Runtime rt = Runtime.getRuntime();

            for (int n : sizes) {
                System.gc();
                double[][] A;
                double[][] B;
                double[][] C;
                try {
                    A = new double[n][n];
                    B = new double[n][n];
                    C = new double[n][n];
                } catch (OutOfMemoryError oom) {
                    csv.printf(Locale.US, "Java,%s,%d,float64,%d,%d,0,NaN,%.1f,%d,NaN,NaN,MEMORY_FAILURE%n",
                            methodLabel, n, threads, warmupRuns, maxBudgetSec, getPeakRssKb());
                    csv.flush();
                    break;
                }

                lcgSeed(42L);
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        A[i][j] = lcgNextDouble();

                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        B[i][j] = lcgNextDouble();

                for (int w = 0; w < warmupRuns; w++) {
                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < n; j++)
                            C[i][j] = 0.0;
                    matmulKernel(A, B, C, n);
                }

                boolean timedOut = false;

                for (int rep = 1; rep <= measuredRuns; rep++) {
                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < n; j++)
                            C[i][j] = 0.0;

                    long tStart = System.nanoTime();
                    matmulKernel(A, B, C, n);
                    long tEnd = System.nanoTime();

                    double elapsed = (tEnd - tStart) * 1e-9;

                    double checksum = 0.0;
                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < n; j++)
                            checksum += C[i][j];

                    long peakRssKb = getPeakRssKb();
                    double usedHeapKb = (rt.totalMemory() - rt.freeMemory()) / 1024.0;
                    String status = (elapsed <= maxBudgetSec) ? "OK" : "TIMEOUT_BUDGET_EXCEEDED";

                    csv.printf(Locale.US, "Java,%s,%d,float64,%d,%d,%d,%.9f,%.1f,%d,%.2f,%.6f,%s%n",
                            methodLabel, n, threads, warmupRuns, rep, elapsed, maxBudgetSec,
                            peakRssKb, usedHeapKb, checksum, status);
                    csv.flush();

                    System.out.printf(Locale.US,
                            "[Java Benchmark] n=%4d | rep=%d/%d | time=%.6f s | PeakRSS=%d kB | Heap=%.1f kB | checksum=%.4f%n",
                            n, rep, measuredRuns, elapsed, peakRssKb, usedHeapKb, checksum);

                    if (elapsed > maxBudgetSec) {
                        timedOut = true;
                        break;
                    }
                }

                if (timedOut) break;
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
    }
}
