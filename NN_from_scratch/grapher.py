import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Folder containing your CSVs
folder = "batch_vs_repeat"  # <-- Replace with your actual folder path

# Match filenames like '30_epochs.csv' and extract the number
def extract_epoch(filename):
    match = re.match(r"(\d+)_epochs\.csv", filename)
    return int(match.group(1)) if match else None

# Collect and sort valid filenames
files = [
    f for f in os.listdir(folder)
    if f.endswith(".csv") and extract_epoch(f) is not None
]
files.sort(key=extract_epoch)

# Prepare data storage
epochs = []
non_batched_means, non_batched_sems = [], []
batched_means, batched_sems = [], []
non_physics_means, non_physics_sems = [], []

# Iterate through sorted files
for file in files:
    epoch = extract_epoch(file)
    epochs.append(epoch)

    df = pd.read_csv(os.path.join(folder, file))

    for column, mean_list, sem_list in [
        ("non_batched_time (s)", non_batched_means, non_batched_sems),
        ("batched_time (s)", batched_means, batched_sems),
        ("non_physics_time (s)", non_physics_means, non_physics_sems)
    ]:
        values = df[column].values
        mean = np.mean(values)
        sem = np.std(values, ddof=1) / np.sqrt(len(values))
        mean_list.append(mean)
        sem_list.append(sem)

# Plotting
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11,
})

capsize = 4

plt.errorbar(epochs, non_batched_means, yerr=non_batched_sems, label=r"\textbf{PINN: Non-Batched}", fmt='-o', capsize=capsize)
plt.errorbar(epochs, batched_means, yerr=batched_sems, label=r"\textbf{PINN: Batched}", fmt='-s', capsize=capsize)
plt.errorbar(epochs, non_physics_means, yerr=non_physics_sems, label=r"\textbf{Non-PINN}", fmt='-^', capsize=capsize)

plt.xlabel(r"\textbf{Epochs}")
plt.ylabel(r"\textbf{Runtime (s)}")
plt.title(r"\textbf{Mean Runtime Across Epochs}")
plt.legend()
plt.savefig('runtimes_pinn.png', dpi=300)
plt.show()
