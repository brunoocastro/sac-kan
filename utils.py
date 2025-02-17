import os

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(x, scores, figure_file):
    running_avg_size = 100
    running_avg = np.zeros(len(scores) - 1)

    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - running_avg_size) : (i + 1)])
    plt.plot(x, running_avg)
    plt.title(f"Running average of previous {running_avg_size} scores")
    plt.savefig(figure_file)
    print(f"Figure saved as {figure_file} with {running_avg_size} running average at {x[-1]} episodes")


def create_folder_if_not_exists(folder):
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exists. Creating now...")
        os.makedirs(folder)
