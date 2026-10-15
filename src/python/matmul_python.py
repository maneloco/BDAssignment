#!/usr/bin/env python3
# src/python/matmul_python.py
import math
import sys
import time
import tracemalloc
from pathlib import Path

_lcg_state = 42


def lcg_seed(seed: int) -> None:
    global _lcg_state
    _lcg_state = seed & 0xFFFFFFFF


def lcg_next_double() -> float:
    global _lcg_state
    _lcg_state = (1664525 * _lcg_state + 1013904223) & 0xFFFFFFFF
    return (_lcg_state / 4294967296.0) * 2.0 - 1.0


def matmul_kernel(A: list[list[float]], B: list[list[float]], C: list[list[float]], n: int) -> None:
    """Pure unoptimized triple-loop O(n^3) matrix multiplication (i-j-k order)."""
    for i in range(n):
        for j in range(n):
            acc = 0.0
            for k in range(n):
                acc += A[i][k] * B[k][j]
            C[i][j] = acc


def get_peak_rss_kb() -> int:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return -1


def validate_from_file(filepath: str) -> bool:
    atol = 1e-9
    rtol = 1e-9

    tokens = Path(filepath).read_text(encoding="utf-8").split()
    idx = 0
    num_tests = int(tokens[idx])
    idx += 1

    for _ in range(num_tests):
        tag = tokens[idx]
        name = tokens[idx + 1]
        n = int(tokens[idx + 2])
        idx += 3

        A = [[0.0] * n for _ in range(n)]
        B = [[0.0] * n for _ in range(n)]
        C_ref = [[0.0] * n for _ in range(n)]
        C_calc = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                A[i][j] = float(tokens[idx])
                idx += 1
        for i in range(n):
            for j in range(n):
                B[i][j] = float(tokens[idx])
                idx += 1
        for i in range(n):
            for j in range(n):
                C_ref[i][j] = float(tokens[idx])
                idx += 1

        if name == "LCG_SEED42_32x32":
            lcg_seed(42)
            for i in range(n):
                for j in range(n):
                    if abs(A[i][j] - lcg_next_double()) > 1e-15:
                        return False
            for i in range(n):
                for j in range(n):
                    if abs(B[i][j] - lcg_next_double()) > 1e-15:
                        return False

        matmul_kernel(A, B, C_calc, n)

        max_diff = 0.0
        for i in range(n):
            for j in range(n):
                diff = abs(C_calc[i][j] - C_ref[i][j])
                if diff > max_diff:
                    max_diff = diff
                if not math.isclose(C_calc[i][j], C_ref[i][j], rel_tol=rtol, abs_tol=atol):
                    return False

        print(f"[Python Validation PASS] {name:<18} (n={n:2d}) | max_abs_error = {max_diff:.3e}")

    return True


def main() -> None:
    val_file = "data/validation_suite.txt"
    csv_file = "results/raw_measurements.csv"
    method_label = "Python_list2D_ijk"

    if not validate_from_file(val_file):
        sys.exit(1)

    sizes = [64, 128, 256, 512, 768, 1024, 1536, 2048]
    warmup_runs = 1
    default_reps = 5
    threads = 1
    max_budget_sec = 450.0  # Allows n=1024 to complete (~150-400s) while capping n>=1536

    tracemalloc.start()

    with open(csv_file, "a", encoding="utf-8") as csv_out:
        budget_exceeded = False

        for n in sizes:
            if budget_exceeded or n > 1024:
                # Document sizes beyond the declared Python time budget without inventing a runtime
                csv_out.write(
                    f"Python,{method_label},{n},float64,{threads},0,0,NaN,{max_budget_sec:.1f},{get_peak_rss_kb()},NaN,NaN,SKIPPED_BUDGET_LIMIT\n"
                )
                csv_out.flush()
                continue

            tracemalloc.reset_peak()

            lcg_seed(42)
            A = [[lcg_next_double() for _ in range(n)] for _ in range(n)]
            B = [[lcg_next_double() for _ in range(n)] for _ in range(n)]
            C = [[0.0] * n for _ in range(n)]

            # Warm-up run for n <= 512 (skip warm-up on very costly runs n >= 768 and document 0 warmups)
            actual_warmups = warmup_runs if n <= 512 else 0
            for _ in range(actual_warmups):
                for i in range(n):
                    for j in range(n):
                        C[i][j] = 0.0
                matmul_kernel(A, B, C, n)

            # Protocol Page 2: "Justify fewer repetitions for costly runs and report the resulting uncertainty."
            reps_to_run = default_reps if n <= 512 else (3 if n == 768 else 3)

            for rep in range(1, reps_to_run + 1):
                for i in range(n):
                    for j in range(n):
                        C[i][j] = 0.0

                t_start = time.perf_counter()
                matmul_kernel(A, B, C, n)
                t_end = time.perf_counter()

                elapsed = t_end - t_start
                checksum = sum(sum(row) for row in C)
                peak_rss_kb = get_peak_rss_kb()
                _, peak_traced_bytes = tracemalloc.get_traced_memory()
                peak_traced_kb = peak_traced_bytes / 1024.0

                status = "OK" if elapsed <= max_budget_sec else "TIMEOUT_BUDGET_EXCEEDED"

                csv_out.write(
                    f"Python,{method_label},{n},float64,{threads},{actual_warmups},{rep},{elapsed:.9f},{max_budget_sec:.1f},{peak_rss_kb},{peak_traced_kb:.2f},{checksum:.6f},{status}\n"
                )
                csv_out.flush()

                print(
                    f"[Python Benchmark] n={n:4d} | rep={rep}/{reps_to_run} | time={elapsed:.6f} s | "
                    f"PeakRSS={peak_rss_kb} kB | Traced={peak_traced_kb:.1f} kB | checksum={checksum:.4f}"
                )

                if elapsed > max_budget_sec:
                    budget_exceeded = True
                    break

    tracemalloc.stop()


if __name__ == "__main__":
    main()
