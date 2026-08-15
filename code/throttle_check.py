"""
Estimates the CPU throttle threshold empirically: runs a sustained stress
workload (default: your naive_matmul ablation config, since that's the one
that hit 79.6C in your data) while continuously sampling `macmon` for CPU
temperature and clock frequency, then looks for the temperature at which
frequency starts dropping under sustained load.

This is a heuristic, not a rigorous proof. It:
  1. Only looks at samples where the CPU was actually busy (usage fraction
     above a threshold), so idle/low-power DVFS dips don't get mistaken
     for thermal throttling.
  2. Buckets busy samples by integer-rounded temperature.
  3. Finds the lowest temperature bucket where the mean frequency in that
     bucket, and every bucket above it, stays meaningfully below the peak
     mean frequency observed across all buckets.

With one run this is suggestive, not conclusive -- treat the reported
threshold as "somewhere around here, worth a closer look," not a precise
number to cite without caveats. Re-running with a longer/hotter workload
(more reps, higher width) will sample more of the high-temperature range
and make the estimate more reliable.

Usage:
    python3 find_throttle_threshold.py \\
        --cmd "./ablate --weights results/cpikan_diffusion_weights/width128 --M 1002001 --warmup 0 --reps 200 --matmul naive --block-size 512" \\
        --sample-interval-ms 1000 \\
        --output-csv throttle_trace.csv \\
        --output-plot throttle_trace.png

Requires: macmon (`brew install macmon`), matplotlib, numpy.
"""

import argparse
import json
import subprocess
import sys
import threading
import time
import csv

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False


def sample_macmon(interval_ms: int, samples: list, stop_event: threading.Event):
    """
    Runs `macmon pipe` continuously in a background thread, appending parsed
    samples (timestamp, temp, pcpu_freq_mhz, pcpu_usage, ecpu_freq_mhz,
    ecpu_usage) to `samples` until stop_event is set.
    """
    proc = subprocess.Popen(
        ["macmon", "pipe", "-i", str(interval_ms)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1
    )
    try:
        for line in proc.stdout:
            if stop_event.is_set():
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            temp = data.get("temp", {}).get("cpu_temp_avg")
            pcpu = data.get("pcpu_usage", [None, None])
            ecpu = data.get("ecpu_usage", [None, None])
            if temp is None:
                continue

            samples.append({
                "t": time.time(),
                "temp": temp,
                "pcpu_freq_mhz": pcpu[0] if len(pcpu) > 0 else None,
                "pcpu_usage": pcpu[1] if len(pcpu) > 1 else None,
                "ecpu_freq_mhz": ecpu[0] if len(ecpu) > 0 else None,
                "ecpu_usage": ecpu[1] if len(ecpu) > 1 else None,
            })
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def run_stress_and_sample(cmd: str, interval_ms: int):
    samples = []
    stop_event = threading.Event()

    sampler_thread = threading.Thread(
        target=sample_macmon, args=(interval_ms, samples, stop_event), daemon=True
    )
    sampler_thread.start()

    print(f"Starting stress workload: {cmd}")
    t_start = time.time()
    workload_proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    workload_proc.wait()
    t_end = time.time()
    print(f"Workload finished after {t_end - t_start:.1f}s. Stopping sampler...")

    # Grab a few extra seconds after the workload ends to see if temp/freq
    # recover -- useful context even though the threshold estimate itself
    # only uses busy-period samples.
    time.sleep(5)
    stop_event.set()
    sampler_thread.join(timeout=10)

    return samples


def estimate_threshold(samples, busy_threshold=0.5, freq_key="pcpu_freq_mhz", usage_key="pcpu_usage",
                        drop_fraction=0.05, sustained_buckets=2):
    """
    Buckets busy samples (usage_key > busy_threshold) by integer temperature,
    computes mean frequency per bucket, and finds the lowest temperature at
    which mean frequency -- and every bucket above it -- stays more than
    `drop_fraction` below the peak mean frequency across all buckets.

    Returns (threshold_temp_or_None, bucket_dict) where bucket_dict maps
    integer temp -> mean frequency, for inspection/plotting.
    """
    busy = [s for s in samples if s.get(usage_key) is not None and s.get(usage_key) >= busy_threshold
            and s.get(freq_key) is not None and s.get("temp") is not None]

    if len(busy) < 10:
        print(f"WARNING: only {len(busy)} busy samples collected (usage >= {busy_threshold}). "
              f"Threshold estimate will be unreliable -- consider a longer/heavier workload.")

    buckets = {}
    for s in busy:
        t_bucket = int(round(s["temp"]))
        buckets.setdefault(t_bucket, []).append(s[freq_key])

    bucket_means = {t: float(np.mean(v)) for t, v in buckets.items() if len(v) > 0}
    if not bucket_means:
        return None, {}

    peak_freq = max(bucket_means.values())
    sorted_temps = sorted(bucket_means.keys())

    threshold = None
    for i, t in enumerate(sorted_temps):
        remaining = sorted_temps[i:]
        if len(remaining) < sustained_buckets:
            break
        # Check whether this temp AND the next (sustained_buckets - 1) temps
        # all show a drop below the peak, so a single noisy low sample
        # doesn't get mistaken for the onset of throttling.
        window = remaining[:sustained_buckets]
        if all(bucket_means[wt] <= peak_freq * (1 - drop_fraction) for wt in window):
            threshold = t
            break

    return threshold, bucket_means


def save_csv(samples, path):
    if not samples:
        print("No samples to save.")
        return
    keys = list(samples[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(samples)
    print(f"Saved {len(samples)} raw samples to {path}")


def save_plot(samples, bucket_means, threshold, path):
    if not HAVE_MATPLOTLIB:
        print("matplotlib not installed -- skipping plot. `pip install matplotlib --break-system-packages`")
        return
    if not samples:
        print("No samples to plot.")
        return

    t0 = samples[0]["t"]
    times = [s["t"] - t0 for s in samples]
    temps = [s["temp"] for s in samples]
    freqs = [s.get("pcpu_freq_mhz") for s in samples]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax0 = axes[0]
    ax0.plot(times, temps, color="tab:red", label="Temp (C)")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Temp (C)", color="tab:red")
    ax0.tick_params(axis="y", labelcolor="tab:red")
    ax1 = ax0.twinx()
    ax1.plot(times, freqs, color="tab:blue", alpha=0.6, label="P-core freq (MHz)")
    ax1.set_ylabel("P-core freq (MHz)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax0.set_title("Temperature and frequency over time")

    ax2 = axes[1]
    if bucket_means:
        temps_sorted = sorted(bucket_means.keys())
        freqs_sorted = [bucket_means[t] for t in temps_sorted]
        ax2.plot(temps_sorted, freqs_sorted, marker="o", color="tab:blue")
        if threshold is not None:
            ax2.axvline(threshold, color="tab:red", linestyle="--",
                        label=f"Estimated threshold: {threshold}C")
            ax2.legend()
    ax2.set_xlabel("Temperature (C, rounded)")
    ax2.set_ylabel("Mean P-core freq during busy samples (MHz)")
    ax2.set_title("Frequency vs. temperature (busy samples only)")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved plot to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", required=True,
                         help="Shell command for the stress workload, e.g. a long-reps ablate run.")
    parser.add_argument("--sample-interval-ms", type=int, default=1000)
    parser.add_argument("--busy-threshold", type=float, default=0.5,
                         help="Only samples with pcpu_usage >= this are treated as 'busy' (default 0.5).")
    parser.add_argument("--drop-fraction", type=float, default=0.05,
                         help="Fractional frequency drop from peak that counts as throttling (default 0.05 = 5%%).")
    parser.add_argument("--sustained-buckets", type=int, default=2,
                         help="Number of consecutive rising-temp buckets that must all show the drop (default 2).")
    parser.add_argument("--output-csv", default="throttle_trace.csv")
    parser.add_argument("--output-plot", default="throttle_trace.png")
    args = parser.parse_args()

    samples = run_stress_and_sample(args.cmd, args.sample_interval_ms)
    print(f"Collected {len(samples)} total samples.")

    threshold, bucket_means = estimate_threshold(
        samples, busy_threshold=args.busy_threshold,
        drop_fraction=args.drop_fraction, sustained_buckets=args.sustained_buckets
    )

    if bucket_means:
        print("\nMean P-core frequency by temperature bucket (busy samples only):")
        for t in sorted(bucket_means.keys()):
            print(f"  {t}C: {bucket_means[t]:.0f} MHz  (n={sum(1 for s in samples if s.get('temp') is not None and int(round(s['temp'])) == t and s.get('pcpu_usage', 0) >= args.busy_threshold)})")

    if threshold is not None:
        print(f"\nEstimated throttle onset: ~{threshold}C "
              f"(frequency drops >= {args.drop_fraction*100:.0f}% below peak, sustained across "
              f"{args.sustained_buckets} consecutive temperature buckets)")
    else:
        print(f"\nNo sustained frequency drop of >= {args.drop_fraction*100:.0f}% detected in this run. "
              f"Either throttling didn't occur, or the workload didn't sustain high-enough temperatures "
              f"long enough to observe it -- try a longer/heavier workload.")

    save_csv(samples, args.output_csv)
    save_plot(samples, bucket_means, threshold, args.output_plot)


if __name__ == "__main__":
    main()
