import numpy as np
from numpy.typing import NDArray
import cv2

from mona_carlo.system import System


def compute_delta_e(image_a: NDArray, image_b: NDArray) -> float:
    """Compute the color difference between two images using the CIEDE2000 color difference formula.

    Parameters
    ----------
    image_a : NDArray
    image_b : NDArray

    Returns
    -------
    float
        Delta E value between the two images.
    """
    # metric for image similarity
    image_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2LAB)
    image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2LAB)

    # compute difference
    diff = cv2.add(image_a, -image_b)

    # separate into L,A,B channel diffs
    diff_L = diff[:, :, 0]
    diff_A = diff[:, :, 1]
    diff_B = diff[:, :, 2]

    # compute delta_e as mean over every pixel using equation from
    # https://en.wikipedia.org/wiki/Color_difference#CIELAB_ΔE*
    return np.mean(np.sqrt(diff_L * diff_L + diff_A * diff_A + diff_B * diff_B))


def compute_system_delta(system_a: System, system_b: System) -> float:
    """Compute the energy delta between two systems.

    Parameters
    ----------
    system_a : System
        _description_
    system_b : System
        _description_

    Returns
    -------
    float
    """
    return compute_delta_e(system_a.state, system_b.state)
