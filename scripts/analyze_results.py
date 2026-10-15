#!/usr/bin/env python3
# scripts/analyze_results.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def iqr(series: pd.Series) -> float:
    return float(np.percentile(series, 75) - np.percentile(series, 25))


def main() -> None:
    raw_path = Path("results/raw_measurements.csv")
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}. Run the benchmarks first.")

    df = pd.read_csv(raw_path)
    valid = df[df["status"].isin(["OK", "TIMEOUT_BUDGET_EXCEEDED"]) & (df["time_sec"] > 0)].copy()

    # 1. Cross-language checksum verification across benchmarked sizes
    print("\n=== Cross-Language Benchmark Checksum Verification ===")
    checksum_summary = (
        valid.groupby(["n", "language"])["checksum"]
        .first()
        .unstack()
    )
    print(checksum_summary.to_string())

    # 2. Compute Median, IQR, Min, Max, and Repetition Count
    summary = (
        valid.groupby(["language", "method", "n"], as_index=False)
        .agg(
            reps=("rep", "count"),
            median_time_s=("time_sec", "median"),
            iqr_time_s=("time_sec", iqr),
            min_time_s=("time_sec", "min"),
            max_time_s=("time_sec", "max"),
            median_peak_rss_kb=("peak_rss_kb", "median"),
            iqr_peak_rss_kb=("peak_rss_kb", iqr),
            median_managed_mem_kb=("managed_mem_kb", "median"),
        )
    )

    # Compute empirical scaling exponent & normalized constant c = T(n) / n^3
    summary["time_per_n3_ns"] = (summary["median_time_s"] * 1e9) / (summary["n"] ** 3)

    summary_out = Path("results/summary_metrics.csv")
    summary.to_csv(summary_out, index=False, float_format="%.9f")
    print(f"\n=== Summary Metrics (Saved to {summary_out}) ===")
    print(summary.to_string(index=False))

    # 3. Generate Plots
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150})
    markers = {"C": "o", "Java": "s", "Python": "^"}

    # Plot A: Log-Log Runtime Scaling with O(n^3) Reference Line
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for lang, group in summary.groupby("language"):
        group = group.sort_values("n")
        ax.errorbar(
            group["n"],
            group["median_time_s"],
            yerr=group["iqr_time_s"] / 2.0,
            label=f"{lang} (median ± IQR/2)",
            marker=markers.get(lang, "o"),
            linewidth=2,
            capsize=4,
        )

    # Reference cubic slope O(n^3) anchored to C at n=256
    c_ref = summary[(summary["language"] == "C") & (summary["n"] == 256)]
    if not c_ref.empty:
        t0 = float(c_ref["median_time_s"].iloc[0])
        n_vals = np.array(sorted(summary["n"].unique()), dtype=float)
        ref_curve = t0 * ((n_vals / 256.0) ** 3)
        ax.plot(n_vals, ref_curve, "--", color="gray", alpha=0.8, label="Reference slope $\\propto n^3$")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Matrix Dimension $n$ ($n \\times n$ square matrix)")
    ax.set_ylabel("Kernel Execution Time (seconds, log scale)")
    ax.set_title("Matrix Multiplication Runtime Scaling Across Languages (Log-Log)")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/log_log_runtime_scaling.png")
    plt.close(fig)

    # Plot B: Linear Scale Runtime Comparison
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for lang, group in summary.groupby("language"):
        group = group.sort_values("n")
        ax.errorbar(
            group["n"],
            group["median_time_s"],
            yerr=group["iqr_time_s"] / 2.0,
            label=f"{lang}",
            marker=markers.get(lang, "o"),
            linewidth=2,
            capsize=4,
        )
    ax.set_xlabel("Matrix Dimension $n$")
    ax.set_ylabel("Median Kernel Execution Time (seconds)")
    ax.set_title("Execution Time vs. Matrix Dimension (Linear Scale)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/linear_runtime_comparison.png")
    plt.close(fig)

    # Plot C: Memory Usage Comparison (Peak RSS in MB)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for lang, group in summary.groupby("language"):
        group = group.sort_values("n")
        ax.plot(
            group["n"],
            group["median_peak_rss_kb"] / 1024.0,
            label=f"{lang} (OS Peak RSS / VmHWM)",
            marker=markers.get(lang, "o"),
            linewidth=2,
        )

    n_vals = np.array(sorted(summary["n"].unique()), dtype=float)
    theoretical_mb = (3.0 * 8.0 * (n_vals ** 2)) / (1024.0 * 1024.0)
    ax.plot(n_vals, theoretical_mb, "--", color="black", alpha=0.7, label="Theoretical $3 \\times 8n^2$ Raw Data (MB)")

    ax.set_xlabel("Matrix Dimension $n$")
    ax.set_ylabel("Memory Footprint (MB)")
    ax.set_title("Peak Process Resident Memory (VmHWM) vs. Matrix Size")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/memory_footprint_comparison.png")
    plt.close(fig)

    print("\n[Analysis Script] Generated plots in results/:")
    print("  - results/log_log_runtime_scaling.png")
    print("  - results/linear_runtime_comparison.png")
    print("  - results/memory_footprint_comparison.png")


if __name__ == "__main__":
    main()
