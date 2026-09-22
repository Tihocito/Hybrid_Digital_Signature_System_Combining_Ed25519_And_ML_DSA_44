import csv
import os
import time
import argparse
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import ed25519
import oqs


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("dataset")
RESULT_FILE = Path("benchmark_results.csv")
RESULTS_DIR = Path("benchmark_results")

# Binary sizes
DATASETS = [
    ("1 KB", 1 * 1024),
    ("10 KB", 10 * 1024),
    ("100 KB", 100 * 1024),
    ("1 MB", 1 * 1024**2),
    ("10 MB", 10 * 1024**2),
    ("100 MB", 100 * 1024**2),
    ("500 MB", 500 * 1024**2),
    ("1 GB", 1 * 1024**3),
]

ML_DSA_ALG = "ML-DSA-44"

ED25519_SIGNATURE_SIZE = 64
ML_DSA_44_SIGNATURE_SIZE = 2420
HYBRID_SIGNATURE_SIZE = (
    ED25519_SIGNATURE_SIZE + ML_DSA_44_SIGNATURE_SIZE
)

RUNS = 100


# ============================================================
# DATASET GENERATION
# ============================================================

def create_dataset_file(path: Path, size: int):
    """Create a deterministic binary file of exactly `size` bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)

    chunk = bytes((i % 256 for i in range(1024 * 1024)))

    with path.open("wb") as f:
        remaining = size
        while remaining > 0:
            n = min(remaining, len(chunk))
            f.write(chunk[:n])
            remaining -= n


def parse_dataset_order(requested_sizes=None):
    available = {label.lower(): (label, size) for label, size in DATASETS}
    if not requested_sizes:
        return DATASETS

    selected = []
    for requested in requested_sizes:
        key = requested.strip().lower()
        if key not in available:
            valid = ", ".join(label for label, _ in DATASETS)
            raise ValueError(f"Unknown dataset size '{requested}'. Use: {valid}")
        selected.append(available[key])
    return selected


def generate_datasets(datasets):
    print("\n=== GENERATING DATASETS ===")

    for label, size in datasets:
        path = DATASET_DIR / f"data_{label.replace(' ', '').replace('KB', 'KB').replace('MB', 'MB').replace('GB', 'GB')}.bin"

        if path.exists() and path.stat().st_size == size:
            print(f"[OK] {label}: already exists ({size:,} bytes)")
            continue

        print(f"[CREATE] {label}: {size:,} bytes")
        start = time.perf_counter()
        create_dataset_file(path, size)
        elapsed = time.perf_counter() - start
        print(f"        created in {elapsed:.3f} s")


# ============================================================
# KEY GENERATION
# ============================================================

def generate_keys():
    print("\n=== GENERATING KEYS ===")

    ed_keygen_times = []
    ml_keygen_times = []

    for _ in range(RUNS):
        start = time.perf_counter()
        ed_private = ed25519.Ed25519PrivateKey.generate()
        ed_public = ed_private.public_key()
        ed_keygen_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        ml_signer = oqs.Signature(ML_DSA_ALG)
        ml_public = ml_signer.generate_keypair()
        ml_keygen_times.append(time.perf_counter() - start)

    ed_keygen_time = calculate_mean(ed_keygen_times)
    ml_keygen_time = calculate_mean(ml_keygen_times)

    # Hybrid uses both key pairs
    hybrid_keygen_time = ed_keygen_time + ml_keygen_time

    print(f"[OK] Ed25519 key pair generated: {ed_keygen_time:.6f} s")
    print(f"[OK] ML-DSA-44 key pair generated: {ml_keygen_time:.6f} s")
    print(f"[OK] Hybrid key pairs generated: {hybrid_keygen_time:.6f} s")

    return (
        ed_private, ed_public,
        ml_signer, ml_public,
        ed_keygen_time, ml_keygen_time, hybrid_keygen_time
    )


# ============================================================
# ED25519
# ============================================================

def benchmark_ed25519(data, private_key, public_key):
    sign_times = []
    verify_times = []

    signature = None

    for _ in range(RUNS):
        start = time.perf_counter()
        signature = private_key.sign(data)
        sign_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        public_key.verify(signature, data)
        verify_times.append(time.perf_counter() - start)

    return (
        calculate_mean(sign_times),
        calculate_mean(verify_times),
        len(signature),
        sign_times,
        verify_times,
    )


# ============================================================
# ML-DSA-44
# ============================================================

def benchmark_ml_dsa(data, signer, public_key):
    sign_times = []
    verify_times = []

    signature = None

    for _ in range(RUNS):
        start = time.perf_counter()
        signature = signer.sign(data)
        sign_times.append(time.perf_counter() - start)

        # A separate verifier object is used for verification.
        verifier = oqs.Signature(ML_DSA_ALG)

        start = time.perf_counter()
        valid = verifier.verify(data, signature, public_key)
        verify_times.append(time.perf_counter() - start)

        if not valid:
            raise RuntimeError("ML-DSA-44 verification failed.")

        # Explicitly release the verifier before the next run.
        try:
            verifier.free()
        except AttributeError:
            pass

    return (
        calculate_mean(sign_times),
        calculate_mean(verify_times),
        len(signature),
        sign_times,
        verify_times,
    )


# ============================================================
# HYBRID
# ============================================================

def benchmark_hybrid(
    data,
    ed_private,
    ed_public,
    ml_signer,
    ml_public,
):
    sign_times = []
    verify_times = []

    hybrid_signature = None

    for _ in range(RUNS):
        # Hybrid signing:
        #   signature = Ed25519 signature || ML-DSA-44 signature
        start = time.perf_counter()

        ed_signature = ed_private.sign(data)
        ml_signature = ml_signer.sign(data)

        hybrid_signature = ed_signature + ml_signature

        sign_times.append(time.perf_counter() - start)

        # Split the hybrid signature.
        ed_signature_received = hybrid_signature[
            :ED25519_SIGNATURE_SIZE
        ]
        ml_signature_received = hybrid_signature[
            ED25519_SIGNATURE_SIZE:
        ]

        # Hybrid verification uses AND composition:
        # both Ed25519 and ML-DSA-44 must verify.
        ml_verifier = oqs.Signature(ML_DSA_ALG)

        start = time.perf_counter()

        try:
            ed_valid = True
            try:
                ed_public.verify(ed_signature_received, data)
            except Exception:
                ed_valid = False

            ml_valid = ml_verifier.verify(
                data,
                ml_signature_received,
                ml_public,
            )

            hybrid_valid = ed_valid and ml_valid

        finally:
            try:
                ml_verifier.free()
            except AttributeError:
                pass

        verify_times.append(time.perf_counter() - start)

        if not hybrid_valid:
            raise RuntimeError("Hybrid verification failed.")

    return (
        calculate_mean(sign_times),
        calculate_mean(verify_times),
        len(hybrid_signature),
        sign_times,
        verify_times,
    )


# ============================================================
# FILE READING
# ============================================================

def load_data(path: Path):
    """Load the complete dataset into memory."""
    with path.open("rb") as f:
        return f.read()


def save_dataset_results(label, rows, raw_rows, keygen_results):
    directory = RESULTS_DIR / label.replace(" ", "_")
    directory.mkdir(parents=True, exist_ok=True)

    result_path = directory / "results.csv"
    with result_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Data Size",
                "Algorithm",
                "Sign Time (s)",
                "Verify Time (s)",
                "Signature Size (bytes)",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    raw_path = directory / "raw_runs.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Data Size",
                "Run",
                "Algorithm",
                "Sign Time (s)",
                "Verify Time (s)",
                "Signature Size (bytes)",
            ],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    keygen_path = directory / "key_generation.csv"
    with keygen_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["Algorithm", "Key Generation Time (s)"],
        )
        writer.writeheader()
        writer.writerows(keygen_results)

    print(f"Dataset results saved to: {directory.resolve()}")


def dataset_result_exists(label):
    directory = RESULTS_DIR / label.replace(" ", "_")
    return all(
        (directory / filename).is_file()
        for filename in ("results.csv", "raw_runs.csv", "key_generation.csv")
    )


def calculate_mean(values):
    if not values:
        raise ValueError("Cannot calculate the mean of an empty sequence")
    return sum(values) / len(values)


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run the hybrid signature benchmark")
    parser.add_argument(
        "--sizes",
        nargs="+",
        metavar="SIZE",
        help="Dataset order, for example: '1 GB' '1 KB' '100 MB'",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip sizes that already have complete per-size result files",
    )
    arguments = parser.parse_args()
    selected_datasets = parse_dataset_order(arguments.sizes)
    print("============================================================")
    print(" HYBRID DIGITAL SIGNATURE BENCHMARK")
    print(" Ed25519 + ML-DSA-44")
    print("============================================================")

    # Check ML-DSA-44 availability
    enabled = oqs.get_enabled_sig_mechanisms()

    if ML_DSA_ALG not in enabled:
        print("\nERROR: ML-DSA-44 is not enabled in liboqs.")
        print("Enabled ML-DSA mechanisms:")
        print([x for x in enabled if "ML-DSA" in x])
        return

    print(f"\n[OK] {ML_DSA_ALG} is available.")
    print(f"[INFO] liboqs version: {oqs.oqs_version()}")
    print(f"[INFO] liboqs-python version: {oqs.oqs_python_version()}")
    print(f"[INFO] Runs per dataset: {RUNS}")

    generate_datasets(selected_datasets)

    (
        ed_private,
        ed_public,
        ml_signer,
        ml_public,
        ed_keygen_time,
        ml_keygen_time,
        hybrid_keygen_time,
    ) = generate_keys()

    results = []

    keygen_results = [
        {"Algorithm": "Ed25519", "Key Generation Time (s)": ed_keygen_time},
        {"Algorithm": "ML-DSA-44", "Key Generation Time (s)": ml_keygen_time},
        {"Algorithm": "Hybrid", "Key Generation Time (s)": hybrid_keygen_time},
    ]

    print("\n=== RUNNING BENCHMARK ===")

    for label, size in selected_datasets:
        if arguments.resume and dataset_result_exists(label):
            print(f"\n--- {label}: already completed, skipping ---")
            continue

        dataset_results = []
        dataset_raw_rows = []
        path = DATASET_DIR / (
            f"data_{label.replace(' ', '')}.bin"
        )

        print(f"\n--- {label} ({size:,} bytes) ---")

        # File reading is intentionally outside the timed
        # cryptographic operations.
        print("[READ] Loading data into memory...")
        read_start = time.perf_counter()
        data = load_data(path)
        read_time = time.perf_counter() - read_start
        print(f"[READ] {read_time:.3f} s")

        # Ed25519
        print("[1/3] Ed25519...")
        ed_sign, ed_verify, ed_size, ed_sign_times, ed_verify_times = benchmark_ed25519(
            data,
            ed_private,
            ed_public,
        )
        print(
            f"      Sign={ed_sign:.6f}s | "
            f"Verify={ed_verify:.6f}s | "
            f"Signature={ed_size} bytes"
        )

        results.append({
            "Data Size": label,
            "Algorithm": "Ed25519",
            "Sign Time (s)": ed_sign,
            "Verify Time (s)": ed_verify,
            "Signature Size (bytes)": ed_size,
        })
        dataset_results.append(results[-1])
        dataset_raw_rows.extend([
            {
                "Data Size": label,
                "Run": run,
                "Algorithm": "Ed25519",
                "Sign Time (s)": sign_time,
                "Verify Time (s)": verify_time,
                "Signature Size (bytes)": ed_size,
            }
            for run, (sign_time, verify_time) in enumerate(zip(ed_sign_times, ed_verify_times), 1)
        ])

        # ML-DSA-44
        print("[2/3] ML-DSA-44...")
        ml_sign, ml_verify, ml_size, ml_sign_times, ml_verify_times = benchmark_ml_dsa(
            data,
            ml_signer,
            ml_public,
        )
        print(
            f"      Sign={ml_sign:.6f}s | "
            f"Verify={ml_verify:.6f}s | "
            f"Signature={ml_size} bytes"
        )

        results.append({
            "Data Size": label,
            "Algorithm": "ML-DSA-44",
            "Sign Time (s)": ml_sign,
            "Verify Time (s)": ml_verify,
            "Signature Size (bytes)": ml_size,
        })
        dataset_results.append(results[-1])
        dataset_raw_rows.extend([
            {
                "Data Size": label,
                "Run": run,
                "Algorithm": "ML-DSA-44",
                "Sign Time (s)": sign_time,
                "Verify Time (s)": verify_time,
                "Signature Size (bytes)": ml_size,
            }
            for run, (sign_time, verify_time) in enumerate(zip(ml_sign_times, ml_verify_times), 1)
        ])

        # Hybrid
        print("[3/3] Hybrid...")
        hy_sign, hy_verify, hy_size, hy_sign_times, hy_verify_times = benchmark_hybrid(
            data,
            ed_private,
            ed_public,
            ml_signer,
            ml_public,
        )
        print(
            f"      Sign={hy_sign:.6f}s | "
            f"Verify={hy_verify:.6f}s | "
            f"Signature={hy_size} bytes"
        )

        results.append({
            "Data Size": label,
            "Algorithm": "Hybrid",
            "Sign Time (s)": hy_sign,
            "Verify Time (s)": hy_verify,
            "Signature Size (bytes)": hy_size,
        })
        dataset_results.append(results[-1])
        dataset_raw_rows.extend([
            {
                "Data Size": label,
                "Run": run,
                "Algorithm": "Hybrid",
                "Sign Time (s)": sign_time,
                "Verify Time (s)": verify_time,
                "Signature Size (bytes)": hy_size,
            }
            for run, (sign_time, verify_time) in enumerate(zip(hy_sign_times, hy_verify_times), 1)
        ])

        save_dataset_results(label, dataset_results, dataset_raw_rows, keygen_results)

    # ========================================================
    # SAVE CSV
    # ========================================================

    with RESULT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Data Size",
                "Algorithm",
                "Sign Time (s)",
                "Verify Time (s)",
                "Signature Size (bytes)",
            ],
        )
        writer.writeheader()

        for row in results:
            writer.writerow(row)

    # ========================================================
    # SAVE KEY-GENERATION RESULTS
    # ========================================================

    keygen_result_file = Path("key_generation_results.csv")

    with keygen_result_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Algorithm", "Key Generation Time (s)"],
        )
        writer.writeheader()
        writer.writerows(keygen_results)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n============================================================")
    print(" BENCHMARK COMPLETED")
    print("============================================================")

    print(f"Dataset directory : {DATASET_DIR.resolve()}")
    print(f"Results CSV       : {RESULT_FILE.resolve()}")
    print(f"Keygen CSV        : {keygen_result_file.resolve()}")

    print("\nKey Generation Time:")
    print(f"Ed25519   : {ed_keygen_time:.6f} s")
    print(f"ML-DSA-44 : {ml_keygen_time:.6f} s")
    print(f"Hybrid    : {hybrid_keygen_time:.6f} s")

    print("\nExpected fixed signature sizes:")
    print(f"Ed25519   : {ED25519_SIGNATURE_SIZE} bytes")
    print(f"ML-DSA-44 : {ML_DSA_44_SIGNATURE_SIZE} bytes")
    print(f"Hybrid    : {HYBRID_SIGNATURE_SIZE} bytes")

    print("\nKey Generation Results:")
    print(f"{'Algorithm':<15}{'Keygen(s)':>15}")
    print("-" * 30)
    for row in keygen_results:
        print(
            f"{row['Algorithm']:<15}"
            f"{row['Key Generation Time (s)']:>15.6f}"
        )

    print("\nResults:")
    print(
        f"{'Data':<10}"
        f"{'Algorithm':<15}"
        f"{'Sign(s)':>12}"
        f"{'Verify(s)':>14}"
        f"{'Sig(bytes)':>14}"
    )

    print("-" * 65)

    for row in results:
        print(
            f"{row['Data Size']:<10}"
            f"{row['Algorithm']:<15}"
            f"{row['Sign Time (s)']:>12.6f}"
            f"{row['Verify Time (s)']:>14.6f}"
            f"{row['Signature Size (bytes)']:>14}"
        )

    # Release ML-DSA signer if supported.
    try:
        ml_signer.free()
    except AttributeError:
        pass


if __name__ == "__main__":
    main()
