import os  # noqa: INP001

import cv2
import numpy as np


def decode_frames(video_path: str) -> list:
    """Decode frames from a video file.

    Args:
    ----
        video_path (str): Path to the video file.

    Returns:
    -------
        list: List of decoded frames (numpy arrays).

    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def sample_frames(frames: list, sample_rate: int = 1) -> list:
    """Sample frames from a list.

    Args:
    ----
        frames (list): List of frames (numpy arrays).
        sample_rate (int, optional): Rate at which to sample frames. Defaults to 1.

    Returns:
    -------
        list: Sampled frames based on the sample_rate.

    """
    return frames[::sample_rate]


def resize_frames(frames: list, target_size: tuple) -> list:
    """Resize frames to a target size.

    Args:
    ----
        frames (list): List of frames (numpy arrays).
        target_size (tuple): Target size for resizing (width, height).

    Returns:
    -------
        list: Resized frames.

    """
    return [cv2.resize(frame, target_size) for frame in frames]


def preprocess_frames(frames: list) -> np.ndarray:
    """Preprocess frames by converting color and normalizing pixel values.

    Args:
    ----
        frames (list): List of frames (numpy arrays).

    Returns:
    -------
        np.ndarray: Processed frames as numpy array with normalized values.

    """
    processed_frames = []
    for frame in frames:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize pixel values to [0, 1]
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        processed_frames.append(frame_normalized)
        # Convert the list of frames to a 4-dimensional numpy array

    processed_frames_array = np.array(processed_frames)
    processed_frames_array = np.expand_dims(
        processed_frames_array, axis=1
    )  # Add channel dimension
    return processed_frames_array  # noqa: RET504


def save_frames_npy(frames: np.ndarray, output_path: str) -> None:
    """Save frames as a .npy file.

    Args:
    ----
        frames (np.ndarray): Frames to save as numpy array.
        output_path (str): Output path for saving the .npy file.

    """
    np.save(output_path, frames)


def preprocess_video(video_path: str, output_dir: str) -> str:
    """Preprocess video by downloading, decoding,sampling, resizing, and saving frames.

    Args:
    ----
        video_path (str): Downloaded YouTube video path.
        output_dir (str): Directory path to save the processed frames.

    Returns:
    -------
        str: Path to the saved .npy file containing processed frames.

    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)  # noqa: PTH103

    # Step 2: Decode frames
    frames = decode_frames(video_path)

    # Step 3: Sample frames (e.g., every 5th frame)
    sampled_frames = sample_frames(frames, sample_rate=5)

    # Step 4: Resize frames
    resized_frames = resize_frames(sampled_frames, target_size=(224, 224))

    # Step 5: Preprocess frames (normalize, convert color)
    processed_frames = preprocess_frames(resized_frames)

    # Step 6: Save frames as .npy file
    output_path = os.path.join(output_dir, "frames.npy")  # noqa: PTH118
    save_frames_npy(processed_frames, output_path)

    return output_path
