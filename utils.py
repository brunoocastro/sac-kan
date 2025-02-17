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


def create_folder_if_not_exists(path):
    """
    Creates folders if they do not exist.
    If a file path is provided, it creates the necessary folders for the file.
    If a folder path is provided, it creates the folder and any necessary parent folders.
    """
    # Check if the path has a file extension
    if os.path.splitext(path)[1]:
        # It's a file path, so get the directory part
        path = os.path.dirname(path)

    # Create the directory if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")
