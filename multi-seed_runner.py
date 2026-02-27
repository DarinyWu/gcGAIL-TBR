"""
Run gcGAIL training over multiple seeds and aggregate results.
"""

import subprocess
import pandas as pd
import pathlib
import time
import json
import matplotlib.pyplot as plt

# ---- CONFIG ----
SEEDS = [0, 1, 2, 3, 4]   # change or extend as needed
BASE_CMD = [
    "python", "gcGAIL_training.py",
    "--no-run_hparam_search",    # reuse fixed params
    "--pretrain_ppo",            # keep pretrain on
    "--total_timesteps", "1638400",
    "--eval_interval", "16384"
]
OUT_DIR = pathlib.Path("outputs/multi_seed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- RUN ----
summaries = []
for s in SEEDS:
    print(f"\n=== Running seed {s} ===")
    t0 = time.time()
    cmd = BASE_CMD + ["--seed", str(s)]
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0

    # find latest gcGAIL output folder
    runs = sorted(pathlib.Path("outputs").glob("gcGAIL_*"), key=lambda p: p.stat().st_mtime)
    latest = runs[-1]
    with open(latest / "reports" / "summary.json", "r") as f:
        summary = json.load(f)
    summary["elapsed_sec"] = elapsed
    summary["run_dir"] = str(latest)
    summary["seed"] = s
    summaries.append(summary)

# ---- AGGREGATE ----
df = pd.DataFrame(summaries)
df.to_csv(OUT_DIR / "multi_seed_summary.csv", index=False)

print("\n==== Aggregate Results ====")
print(df[[
    "seed",
    "training_mean_return_after",
    "classification_accuracy",
    "elapsed_sec"
]])

# Plot distribution of returns and accuracy
plt.figure(figsize=(7,4))
plt.errorbar(df["seed"], df["training_mean_return_after"], fmt="o-", label="Mean return after")
plt.xlabel("Seed")
plt.ylabel("Return")
plt.title("gcGAIL performance across seeds")
plt.tight_layout()
plt.savefig(OUT_DIR / "returns_across_seeds.png", dpi=300)
plt.close()

plt.figure(figsize=(7,4))
plt.errorbar(df["seed"], df["classification_accuracy"], fmt="s-", label="Test accuracy")
plt.xlabel("Seed")
plt.ylabel("Accuracy")
plt.title("gcGAIL test accuracy across seeds")
plt.tight_layout()
plt.savefig(OUT_DIR / "accuracy_across_seeds.png", dpi=300)
plt.close()

plt.figure(figsize=(7,4))
plt.bar(df["seed"].astype(str), df["elapsed_sec"], color="steelblue")
plt.xlabel("Seed")
plt.ylabel("Runtime (seconds)")
plt.title("gcGAIL runtime across seeds")
plt.tight_layout()
plt.savefig(OUT_DIR / "runtime_across_seeds.png", dpi=300)
plt.close()