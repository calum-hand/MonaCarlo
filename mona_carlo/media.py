from typing import List, Iterable

import cv2
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


def numpy_to_image(filename: str, *arrays: Iterable[NDArray]):
    """Write numpy array(s) to an image file.
    If multiple arrays are provided, they will be concatenated horizontally before writing as single image.

    Parameters
    ----------
    filename : str
        Destination file path.
    """
    cv2.imwrite(filename, np.hstack(arrays))


def images_to_video(
    image_paths: List[str], file_name: str = "output.mp4", fps: int = 1_000
):
    """Convert a list of images to a video file.
    All images MUST be the same size.

    Parameters
    ----------
    image_paths : List[str]
        Paths to images in order of appearance in the video.
    file_name : str, optional
        destination file path, by default "output.mp4"
    fps : int, optional
        frames per second, by default 1_000
    """
    # Read the first image to get dimensions
    frame = cv2.imread(image_paths[0])
    height, width, _ = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or use 'XVID' for .avi
    video = cv2.VideoWriter(file_name, fourcc, fps, (width, height))  # 30 is the fps

    for image_path in tqdm(image_paths):
        frame = cv2.imread(image_path)
        video.write(frame)
    video.release()
