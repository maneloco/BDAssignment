#!/usr/bin/env bash
# run_all.sh
set -euo pipefail

mkdir -p data results build/c build/java

ENV_FILE="results/environment_info.txt"
echo "=== Hardware & Software Environment Specification ===" > "$ENV_FILE"
echo "Timestamp: $(date -Is)" >> "$ENV_FILE"
echo -e "\n--- OS & Kernel ---" >> "$ENV_FILE"
uname -a >> "$ENV_FILE"
echo -e "\n--- CPU Topology ---" >> "$ENV_FILE"
lscpu | grep -E "Model name|Socket|Thread|NUMA|Core|CPU\(s\)|MHz|L1d|L2|L3" >> "$ENV_FILE" || true
echo -e "\n--- RAM ---" >> "$ENV_FILE"
free -h >> "$ENV_FILE"
echo -e "\n--- Toolchains & Runtimes ---" >> "$ENV_FILE"
gcc --version | head -n 1 >> "$ENV_FILE"
java -version >> "$ENV_FILE" 2>&1
python3 --version >> "$ENV_FILE"
echo " Environment details logged to $ENV_FILE"

# Initialize CSV header
CSV_FILE="results/raw_measurements.csv"
echo "language,method,n,numeric_type,threads,warmup_runs,rep,time_sec,budget_sec,peak_rss_kb,managed_mem_kb,checksum,status" > "$CSV_FILE"

echo -e "\n=== 1. Compiling and Running C Reference Generator ==="
gcc -O2 -std=c11 -Wall -Wextra src/c/generate_and_validate.c -o build/c/gen_val -lm
./build/c/gen_val data/validation_suite.txt

echo -e "\n=== 2. Compiling and Benchmarking C Implementation (-O2) ==="
gcc -O2 -std=c11 -Wall -Wextra src/c/matmul_c.c -o build/c/matmul_c_O2 -lm
./build/c/matmul_c_O2 C_O2_row_major_1D

echo -e "\n=== 3. Compiling and Benchmarking Java Implementation ==="
javac -d build/java src/java/MatMulBenchmark.java
java -Xms512m -Xmx4g -cp build/java MatMulBenchmark

echo -e "\n=== 4. Validating and Benchmarking Python Implementation ==="
python3 src/python/matmul_python.py

echo -e "\n=== 5. Running Statistical Summary and Generating Plots ==="
python3 scripts/analyze_results.py

echo -e "\n=== All Benchmarks Complete! ==="
