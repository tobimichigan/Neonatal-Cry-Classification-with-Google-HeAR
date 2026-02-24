import matplotlib.pyplot as plt
import numpy as np
from 01_config import CLASS_NAMES, PLOTS_DIR

def plot_class_distribution(labels, split_name):
    counts = [sum(np.array(labels)==i) for i in range(len(CLASS_NAMES))]
    plt.figure(figsize=(6,4))
    plt.bar(CLASS_NAMES, counts)
    plt.title(f"{split_name} Class Distribution")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/dist_{split_name}.png")
    plt.close()
