import os
import csv
import time
import gc
import base64
import argparse
import threading
import psutil
import oqs

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset"

RUNS = 100

RAW_RESULT_FILE = "benchmark_runs.csv"
SUMMARY_RESULT_FILE = "benchmark_summary.csv"
RESULTS_DIR = "benchmark_results"

DATASETS = [
    ("1 KB", 1 * 1024),
    ("1 MB", 1 * 1024 * 1024),
    ("100 MB", 100 * 1024 * 1024),
    ("1 GB", 1 * 1024 * 1024 * 1024),
]


# ============================================================
# PROCESS / MEMORY
# ============================================================

PROCESS = psutil.Process(os.getpid())


def get_memory_mb():
    """
    Current process RSS memory in MB.
    """
    return PROCESS.memory_info().rss / (1024 * 1024)


class MemoryMonitor:
    """
    Continuously monitors process RSS during one benchmark.
    """

    def __init__(self, interval=0.001):

        self.interval = interval
        self.running = False
        self.peak_mb = 0.0
        self.thread = None

    def _monitor(self):

        while self.running:

            current = get_memory_mb()

            if current > self.peak_mb:
                self.peak_mb = current

            time.sleep(self.interval)

    def start(self):

        self.peak_mb = get_memory_mb()

        self.running = True

        self.thread = threading.Thread(
            target=self._monitor,
            daemon=True
        )

        self.thread.start()

    def stop(self):

        self.running = False

        if self.thread is not None:
            self.thread.join()

        # Final memory check
        current = get_memory_mb()

        if current > self.peak_mb:
            self.peak_mb = current

        return self.peak_mb


# ============================================================
# CREATE DATASET
# ============================================================

def parse_dataset_order(requested_sizes=None):
    available = {name.lower(): (name, size) for name, size in DATASETS}
    if not requested_sizes:
        return DATASETS

    selected = []
    for requested in requested_sizes:
        key = requested.strip().lower()
        if key not in available:
            valid = ", ".join(name for name, _ in DATASETS)
            raise ValueError(f"Unknown dataset size '{requested}'. Use: {valid}")
        selected.append(available[key])
    return selected


def create_dataset(datasets):

    os.makedirs(DATASET_DIR, exist_ok=True)

    print("\nCreating dataset...\n")

    for name, size in datasets:

        filename = name.replace(" ", "_")

        path = os.path.join(
            DATASET_DIR,
            f"{filename}.bin"
        )

        if (
            os.path.exists(path)
            and os.path.getsize(path) == size
        ):

            print(f"[OK] {name}")
            continue

        print(
            f"[CREATE] {name} -> "
            f"{size:,} bytes"
        )

        with open(path, "wb") as f:

            chunk_size = 1024 * 1024

            remaining = size

            while remaining > 0:

                current = min(
                    chunk_size,
                    remaining
                )

                f.write(os.urandom(current))

                remaining -= current

        print(f"[DONE] {path}")

    print("\nDataset creation completed.\n")


# ============================================================
# KEY GENERATION
# ============================================================

def generate_ed25519_keypair():

    private_key = Ed25519PrivateKey.generate()

    public_key = private_key.public_key()

    return private_key, public_key


def generate_ml_dsa_keypair():

    signer = oqs.Signature(
        "ML-DSA-44"
    )

    public_key = signer.generate_keypair()

    secret_key = signer.export_secret_key()

    return public_key, secret_key


# ============================================================
# BENCHMARK KEY GENERATION
# ============================================================

def benchmark_key_generation():

    ed_times = []
    ml_times = []
    hybrid_times = []

    print("\n" + "=" * 70)
    print(f"KEY GENERATION - {RUNS} RUNS")
    print("=" * 70)

    for run in range(1, RUNS + 1):

        # ----------------------------------------------------
        # Ed25519
        # ----------------------------------------------------

        start = time.perf_counter()

        ed_private, ed_public = (
            generate_ed25519_keypair()
        )

        ed_time = time.perf_counter() - start


        # ----------------------------------------------------
        # ML-DSA-44
        # ----------------------------------------------------

        start = time.perf_counter()

        ml_public, ml_secret = (
            generate_ml_dsa_keypair()
        )

        ml_time = time.perf_counter() - start


        # ----------------------------------------------------
        # Hybrid
        # ----------------------------------------------------

        start = time.perf_counter()

        hybrid_ed_private, hybrid_ed_public = (
            generate_ed25519_keypair()
        )

        hybrid_ml_public, hybrid_ml_secret = (
            generate_ml_dsa_keypair()
        )

        hybrid_time = time.perf_counter() - start


        ed_times.append(ed_time)
        ml_times.append(ml_time)
        hybrid_times.append(hybrid_time)


        print(
            f"Run {run:02d} | "
            f"Ed25519: {ed_time * 1000:.6f} ms | "
            f"ML-DSA-44: {ml_time * 1000:.6f} ms | "
            f"Hybrid: {hybrid_time * 1000:.6f} ms"
        )


        del ed_private
        del ed_public

        del ml_public
        del ml_secret

        del hybrid_ed_private
        del hybrid_ed_public
        del hybrid_ml_public
        del hybrid_ml_secret

        gc.collect()


    ed_mean = sum(ed_times) / len(ed_times)
    ml_mean = sum(ml_times) / len(ml_times)
    hybrid_mean = sum(hybrid_times) / len(hybrid_times)


    print("\nKEY GENERATION MEAN")
    print(
        f"Ed25519     : {ed_mean * 1000:.6f} ms"
    )
    print(
        f"ML-DSA-44   : {ml_mean * 1000:.6f} ms"
    )
    print(
        f"Hybrid      : {hybrid_mean * 1000:.6f} ms"
    )


    return {
        "Ed25519": ed_mean,
        "ML-DSA-44": ml_mean,
        "Hybrid": hybrid_mean
    }


# ============================================================
# CREATE BENCHMARK KEYS
# ============================================================

def create_benchmark_keys():

    print("\nGenerating benchmark keys...")

    # Ed25519
    ed_private_key, ed_public_key = (
        generate_ed25519_keypair()
    )

    # ML-DSA-44
    ml_public_key, ml_secret_key = (
        generate_ml_dsa_keypair()
    )

    print("Benchmark keys generated successfully.\n")

    return (
        ed_private_key,
        ed_public_key,
        ml_public_key,
        ml_secret_key
    )


# ============================================================
# ED25519
# ============================================================

def benchmark_ed25519(
    data,
    ed_private_key,
    ed_public_key
):

    memory_monitor = MemoryMonitor()

    memory_monitor.start()


    # --------------------------------------------------------
    # Signing
    # --------------------------------------------------------

    start = time.perf_counter()

    signature = ed_private_key.sign(data)

    sign_time = time.perf_counter() - start


    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    start = time.perf_counter()

    try:

        ed_public_key.verify(
            signature,
            data
        )

        valid = True

    except Exception:

        valid = False


    verify_time = time.perf_counter() - start


    peak_ram = memory_monitor.stop()


    if not valid:

        raise RuntimeError(
            "Ed25519 verification failed"
        )


    return (
        sign_time,
        verify_time,
        peak_ram,
        len(signature)
    )


# ============================================================
# ML-DSA-44
# ============================================================

def benchmark_ml_dsa(
    data,
    ml_public_key,
    ml_secret_key
):

    memory_monitor = MemoryMonitor()

    memory_monitor.start()


    # --------------------------------------------------------
    # Signing
    # --------------------------------------------------------

    signer = oqs.Signature(
        "ML-DSA-44",
        ml_secret_key
    )

    start = time.perf_counter()

    signature = signer.sign(data)

    sign_time = time.perf_counter() - start


    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    verifier = oqs.Signature(
        "ML-DSA-44"
    )

    start = time.perf_counter()

    valid = verifier.verify(
        data,
        signature,
        ml_public_key
    )

    verify_time = time.perf_counter() - start


    peak_ram = memory_monitor.stop()


    if not valid:

        raise RuntimeError(
            "ML-DSA-44 verification failed"
        )


    return (
        sign_time,
        verify_time,
        peak_ram,
        len(signature)
    )


# ============================================================
# HYBRID
# ============================================================

def benchmark_hybrid(
    data,
    ed_private_key,
    ed_public_key,
    ml_public_key,
    ml_secret_key
):

    memory_monitor = MemoryMonitor()

    memory_monitor.start()


    # ========================================================
    # ED25519 SIGNING
    # ========================================================

    start = time.perf_counter()

    ed_signature = ed_private_key.sign(data)

    ed_sign_time = time.perf_counter() - start


    # ========================================================
    # ML-DSA-44 SIGNING
    # ========================================================

    signer = oqs.Signature(
        "ML-DSA-44",
        ml_secret_key
    )

    start = time.perf_counter()

    ml_signature = signer.sign(data)

    ml_sign_time = time.perf_counter() - start


    # Total hybrid signing time
    sign_time = (
        ed_sign_time
        + ml_sign_time
    )


    # ========================================================
    # HYBRID SIGNATURE
    # ========================================================

    hybrid_signature = (
        ed_signature
        + ml_signature
    )


    # ========================================================
    # ED25519 VERIFICATION
    # ========================================================

    start = time.perf_counter()

    try:

        ed_public_key.verify(
            ed_signature,
            data
        )

        ed_valid = True

    except Exception:

        ed_valid = False


    ed_verify_time = (
        time.perf_counter()
        - start
    )


    # ========================================================
    # ML-DSA-44 VERIFICATION
    # ========================================================

    verifier = oqs.Signature(
        "ML-DSA-44"
    )

    start = time.perf_counter()

    ml_valid = verifier.verify(
        data,
        ml_signature,
        ml_public_key
    )

    ml_verify_time = (
        time.perf_counter()
        - start
    )


    # Total hybrid verification time
    verify_time = (
        ed_verify_time
        + ml_verify_time
    )


    # ========================================================
    # AND COMPOSITION
    # ========================================================

    valid = (
        ed_valid
        and ml_valid
    )


    peak_ram = memory_monitor.stop()


    if not valid:

        raise RuntimeError(
            "Hybrid verification failed"
        )


    return (
        sign_time,
        verify_time,
        peak_ram,
        len(hybrid_signature)
    )


# ============================================================
# MEAN
# ============================================================

def calculate_mean(values):
    if not values:
        raise ValueError("Cannot calculate the mean of an empty sequence")
    return sum(values) / len(values)


# ============================================================
# BENCHMARK ONE DATASET
# ============================================================

def benchmark_dataset(
    name,
    path,
    keys,
    raw_results
):

    (
        ed_private_key,
        ed_public_key,
        ml_public_key,
        ml_secret_key
    ) = keys


    print("\n" + "=" * 70)
    print(f"DATA SIZE: {name}")
    print("=" * 70)


    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("Loading data...")

    with open(path, "rb") as f:

        data = f.read()


    print(
        f"Loaded {len(data):,} bytes."
    )


    # --------------------------------------------------------
    # Store measurements for each run.
    # --------------------------------------------------------

    ed_sign_times = []
    ed_verify_times = []
    ed_ram = []

    ml_sign_times = []
    ml_verify_times = []
    ml_ram = []

    hybrid_sign_times = []
    hybrid_verify_times = []
    hybrid_ram = []


    ed_signature_size = None
    ml_signature_size = None
    hybrid_signature_size = None


    # ========================================================
    # RUN BENCHMARK
    # ========================================================

    for run in range(1, RUNS + 1):

        print(
            f"\n--- Run {run}/{RUNS} ---"
        )


        # ====================================================
        # ED25519
        # ====================================================

        (
            ed_sign,
            ed_verify,
            ed_memory,
            ed_sig_size
        ) = benchmark_ed25519(
            data,
            ed_private_key,
            ed_public_key
        )

        ed_sign_times.append(ed_sign)
        ed_verify_times.append(ed_verify)
        ed_ram.append(ed_memory)

        ed_signature_size = ed_sig_size


        print(
            f"Ed25519     | "
            f"Sign: {ed_sign:.6f} s | "
            f"Verify: {ed_verify:.6f} s | "
            f"RAM: {ed_memory:.2f} MB"
        )


        # ====================================================
        # ML-DSA-44
        # ====================================================

        (
            ml_sign,
            ml_verify,
            ml_memory,
            ml_sig_size
        ) = benchmark_ml_dsa(
            data,
            ml_public_key,
            ml_secret_key
        )

        ml_sign_times.append(ml_sign)
        ml_verify_times.append(ml_verify)
        ml_ram.append(ml_memory)

        ml_signature_size = ml_sig_size


        print(
            f"ML-DSA-44   | "
            f"Sign: {ml_sign:.6f} s | "
            f"Verify: {ml_verify:.6f} s | "
            f"RAM: {ml_memory:.2f} MB"
        )


        # ====================================================
        # HYBRID
        # ====================================================

        (
            hybrid_sign,
            hybrid_verify,
            hybrid_memory,
            hybrid_sig_size
        ) = benchmark_hybrid(
            data,
            ed_private_key,
            ed_public_key,
            ml_public_key,
            ml_secret_key
        )

        hybrid_sign_times.append(
            hybrid_sign
        )

        hybrid_verify_times.append(
            hybrid_verify
        )

        hybrid_ram.append(
            hybrid_memory
        )

        hybrid_signature_size = (
            hybrid_sig_size
        )


        print(
            f"Hybrid      | "
            f"Sign: {hybrid_sign:.6f} s | "
            f"Verify: {hybrid_verify:.6f} s | "
            f"RAM: {hybrid_memory:.2f} MB"
        )

        raw_results.extend([
            [name, run, "Ed25519", ed_sign, ed_verify, ed_memory],
            [name, run, "ML-DSA-44", ml_sign, ml_verify, ml_memory],
            [name, run, "Hybrid", hybrid_sign, hybrid_verify, hybrid_memory],
        ])


    # ========================================================
    # CALCULATE MEAN
    # ========================================================

    ed_sign_mean = calculate_mean(
        ed_sign_times
    )

    ed_verify_mean = calculate_mean(
        ed_verify_times
    )

    ed_ram_mean = calculate_mean(
        ed_ram
    )


    ml_sign_mean = calculate_mean(
        ml_sign_times
    )

    ml_verify_mean = calculate_mean(
        ml_verify_times
    )

    ml_ram_mean = calculate_mean(
        ml_ram
    )


    hybrid_sign_mean = calculate_mean(
        hybrid_sign_times
    )

    hybrid_verify_mean = calculate_mean(
        hybrid_verify_times
    )

    hybrid_ram_mean = calculate_mean(
        hybrid_ram
    )


    # ========================================================
    # THROUGHPUT
    # ========================================================

    size_mb = len(data) / (
        1024 * 1024
    )


    ed_throughput = (
        size_mb / ed_sign_mean
    )

    ml_throughput = (
        size_mb / ml_sign_mean
    )

    hybrid_throughput = (
        size_mb / hybrid_sign_mean
    )


    # ========================================================
    # BASE64 SIGNATURE SIZE
    # ========================================================

    ed_base64_size = len(
        base64.b64encode(
            b"\x00" * ed_signature_size
        )
    )

    ml_base64_size = len(
        base64.b64encode(
            b"\x00" * ml_signature_size
        )
    )

    hybrid_base64_size = len(
        base64.b64encode(
            b"\x00" * hybrid_signature_size
        )
    )


    # ========================================================
    # PRINT MEAN
    # ========================================================

    print("\n" + "-" * 70)
    print(f"MEAN RESULTS - {name}")
    print("-" * 70)


    print(
        f"Ed25519     | "
        f"Sign: {ed_sign_mean:.6f} s | "
        f"Verify: {ed_verify_mean:.6f} s | "
        f"RAM: {ed_ram_mean:.2f} MB | "
        f"Throughput: {ed_throughput:.2f} MB/s"
    )


    print(
        f"ML-DSA-44   | "
        f"Sign: {ml_sign_mean:.6f} s | "
        f"Verify: {ml_verify_mean:.6f} s | "
        f"RAM: {ml_ram_mean:.2f} MB | "
        f"Throughput: {ml_throughput:.2f} MB/s"
    )


    print(
        f"Hybrid      | "
        f"Sign: {hybrid_sign_mean:.6f} s | "
        f"Verify: {hybrid_verify_mean:.6f} s | "
        f"RAM: {hybrid_ram_mean:.2f} MB | "
        f"Throughput: {hybrid_throughput:.2f} MB/s"
    )


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "Ed25519": {
            "Sign": ed_sign_mean,
            "Verify": ed_verify_mean,
            "RAM": ed_ram_mean,
            "Signature": ed_signature_size,
            "Throughput": ed_throughput,
            "PublicKey": 32,
            "Base64": ed_base64_size
        },

        "ML-DSA-44": {
            "Sign": ml_sign_mean,
            "Verify": ml_verify_mean,
            "RAM": ml_ram_mean,
            "Signature": ml_signature_size,
            "Throughput": ml_throughput,
            "PublicKey": 1312,
            "Base64": ml_base64_size
        },

        "Hybrid": {
            "Sign": hybrid_sign_mean,
            "Verify": hybrid_verify_mean,
            "RAM": hybrid_ram_mean,
            "Signature": hybrid_signature_size,
            "Throughput": hybrid_throughput,
            "PublicKey": 1344,
            "Base64": hybrid_base64_size
        }
    }


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(
    results,
    keygen_results,
    datasets
):

    with open(
        SUMMARY_RESULT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Metric",
            "Ed25519",
            "ML-DSA-44",
            "Hybrid"
        ])


        # ----------------------------------------------------
        # Key Generation
        # ----------------------------------------------------

        writer.writerow([
            "KeyGen (ms)",
            keygen_results["Ed25519"] * 1000,
            keygen_results["ML-DSA-44"] * 1000,
            keygen_results["Hybrid"] * 1000
        ])


        # ----------------------------------------------------
        # Requested dataset sizes
        # ----------------------------------------------------

        for size_name, _ in datasets:

            data = results[size_name]


            writer.writerow([
                f"Sign {size_name} (s)",
                data["Ed25519"]["Sign"],
                data["ML-DSA-44"]["Sign"],
                data["Hybrid"]["Sign"]
            ])


            writer.writerow([
                f"Verify {size_name} (s)",
                data["Ed25519"]["Verify"],
                data["ML-DSA-44"]["Verify"],
                data["Hybrid"]["Verify"]
            ])


        # ----------------------------------------------------
        # Peak RAM
        # ----------------------------------------------------

        largest_size_name = max(datasets, key=lambda item: item[1])[0]
        data = results[largest_size_name]

        writer.writerow([
            "Peak RAM (MB)",
            data["Ed25519"]["RAM"],
            data["ML-DSA-44"]["RAM"],
            data["Hybrid"]["RAM"]
        ])


        # ----------------------------------------------------
        # Signature size
        # ----------------------------------------------------

        first_size_name = datasets[0][0]
        data = results[first_size_name]

        writer.writerow([
            "Signature size (B)",
            data["Ed25519"]["Signature"],
            data["ML-DSA-44"]["Signature"],
            data["Hybrid"]["Signature"]
        ])


        # ----------------------------------------------------
        # Public key
        # ----------------------------------------------------

        writer.writerow([
            "Public key (B)",
            data["Ed25519"]["PublicKey"],
            data["ML-DSA-44"]["PublicKey"],
            data["Hybrid"]["PublicKey"]
        ])


        # ----------------------------------------------------
        # Base64 signature
        # ----------------------------------------------------

        writer.writerow([
            "Base64 signature (B)",
            data["Ed25519"]["Base64"],
            data["ML-DSA-44"]["Base64"],
            data["Hybrid"]["Base64"]
        ])


        # ----------------------------------------------------
        # Throughput
        # ----------------------------------------------------

        data = results[largest_size_name]

        writer.writerow([
            "Throughput (MB/s)",
            data["Ed25519"]["Throughput"],
            data["ML-DSA-44"]["Throughput"],
            data["Hybrid"]["Throughput"]
        ])


    print(
        f"\nSummary saved to: "
        f"{SUMMARY_RESULT_FILE}"
    )


def save_dataset_results(name, result, raw_results, keygen_results):
    directory = os.path.join(RESULTS_DIR, name.replace(" ", "_"))
    os.makedirs(directory, exist_ok=True)

    summary_path = os.path.join(directory, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Ed25519", "ML-DSA-44", "Hybrid"])
        writer.writerow([
            "KeyGen (ms)",
            keygen_results["Ed25519"] * 1000,
            keygen_results["ML-DSA-44"] * 1000,
            keygen_results["Hybrid"] * 1000,
        ])
        for metric in ("Sign", "Verify"):
            writer.writerow([
                f"{metric} {name} (s)",
                result["Ed25519"][metric],
                result["ML-DSA-44"][metric],
                result["Hybrid"][metric],
            ])
        writer.writerow([
            "Peak RAM (MB)",
            result["Ed25519"]["RAM"],
            result["ML-DSA-44"]["RAM"],
            result["Hybrid"]["RAM"],
        ])
        writer.writerow([
            "Throughput (MB/s)",
            result["Ed25519"]["Throughput"],
            result["ML-DSA-44"]["Throughput"],
            result["Hybrid"]["Throughput"],
        ])

    raw_path = os.path.join(directory, "raw_runs.csv")
    with open(raw_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Data Size", "Run", "Algorithm", "Sign Time (s)",
            "Verify Time (s)", "Peak RAM (MB)",
        ])
        writer.writerows(raw_results)

    keygen_path = os.path.join(directory, "key_generation.csv")
    with open(keygen_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Algorithm", "Key Generation Time (s)"])
        for algorithm, value in keygen_results.items():
            writer.writerow([algorithm, value])

    print(f"Dataset results saved to: {directory}")


# ============================================================
# SAVE ALL RUNS
# ============================================================

def save_raw_runs(
    raw_results
):

    with open(
        RAW_RESULT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Data Size",
            "Run",
            "Algorithm",
            "Sign Time (s)",
            "Verify Time (s)",
            "Peak RAM (MB)"
        ])


        for row in raw_results:

            writer.writerow(row)


    print(f"Raw {RUNS}-run results saved to: {RAW_RESULT_FILE}")


# ============================================================
# MAIN BENCHMARK
# ============================================================

def run_benchmark(datasets):

    print("=" * 70)
    print(f"{RUNS}-RUN HYBRID DIGITAL SIGNATURE BENCHMARK")
    print("=" * 70)

    print(
        f"\nEach measurement will be repeated "
        f"{RUNS} times."
    )


    # ========================================================
    # KEY GENERATION
    # ========================================================

    keygen_results = (
        benchmark_key_generation()
    )


    # ========================================================
    # CREATE KEYS FOR SIGN/VERIFY
    # ========================================================

    keys = create_benchmark_keys()


    # ========================================================
    # DATASET BENCHMARK
    # ========================================================

    results = {}

    raw_results = []


    for name, size in datasets:

        filename = name.replace(
            " ",
            "_"
        )

        path = os.path.join(
            DATASET_DIR,
            f"{filename}.bin"
        )


        result = benchmark_dataset(name, path, keys, raw_results)


        results[name] = result

        dataset_raw_results = [row for row in raw_results if row[0] == name]
        save_dataset_results(name, result, dataset_raw_results, keygen_results)


    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    save_summary(
        results,
        keygen_results,
        datasets
    )

    save_raw_runs(raw_results)


    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETED")
    print("=" * 70)

    print(
        f"\nSummary: "
        f"{SUMMARY_RESULT_FILE}"
    )

    print(
        f"Raw runs: "
        f"{RAW_RESULT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hybrid signature benchmark")
    parser.add_argument(
        "--sizes",
        nargs="+",
        metavar="SIZE",
        help="Dataset order, for example: '1 GB' '1 KB' '100 MB' '1 MB'",
    )
    arguments = parser.parse_args()
    selected_datasets = parse_dataset_order(arguments.sizes)

    create_dataset(selected_datasets)
    run_benchmark(selected_datasets)
