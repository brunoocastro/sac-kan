import os
from typing import List

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(x, scores, figure_file):
    running_avg_size = 100
    running_avg = np.zeros(len(scores))

    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - running_avg_size) : (i + 1)])
    plt.plot(x, running_avg)
    plt.title(f"Running average of previous {running_avg_size} scores")
    plt.savefig(figure_file)
    # print(f"Figure saved as {figure_file} with {running_avg_size} running average at {x[-1]} episodes")


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


def frames_to_gif(frames: List[np.ndarray], output_path: str, metadata: dict = {}, fps: int = 30) -> None:
    """
    Converts RGB frames (numpy arrays) to a GIF file and adds step counter and metadata

    Args:
        frames: List of numpy arrays (RGB)
        output_path: Path to save the GIF
        metadata: Dictionary with metadata to display (default: empty)
        fps: Frames per second (default: 30)
    """
    # Convert frames to numpy arrays if needed
    frames = [np.array(frame) if frame is not None else np.zeros((400, 600, 3), dtype=np.uint8) for frame in frames]

    # Ensure the extension is .gif
    if not output_path.endswith('.gif'):
        output_path = output_path.replace('.mp4', '.gif')

    # Calculate fixed positions
    frame_width = frames[0].shape[1]
    step_x = frame_width - 150  # Fixed position from right for step counter
    metadata_x = 10  # Fixed position from left for metadata
    base_y = 30  # Starting Y position

    # Add step counter and metadata to each frame
    frames_with_steps = []
    for i, frame in enumerate(frames):
        frame_with_text = frame.copy()

        # Add metadata (left side)
        if metadata:
            current_y = base_y
            for key, value in metadata.items():
                # Limit value if is numeric
                if isinstance(value, (int, float)):
                    value = f"{value:.2f}"

                text = f'{key}: {value}'
                cv2.putText(
                    frame_with_text,
                    text,
                    (metadata_x, current_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                current_y += 30  # Vertical spacing between metadata lines

        # Add step counter (right side)
        step_text = f'Step: {i}'
        cv2.putText(
            frame_with_text, step_text, (step_x, base_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
        )

        frames_with_steps.append(frame_with_text)

    # Save as GIF using imageio
    imageio.mimsave(output_path, frames_with_steps, fps=fps, format='GIF')
