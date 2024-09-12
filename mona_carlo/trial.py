from typing import Any

import numpy as np
from numpy.typing import NDArray

from mona_carlo.system import System
from mona_carlo.sampler import random_sampler


def sample_swap_value(system: System, active_component: Any) -> Any:
    """Selects a component from the system to swap with the active component.
    The selection is weighted by the occurance of components in the system.
    The active_component cannot be selected.
    
    Parameters
    ----------
    system : System
    active_component : Any

    Returns
    -------
    Any
        Selected component to swap with the active component.
    """
    value_mapping = system.components
    possible_values, counts = [], []

    for k in value_mapping:
        if (k != active_component).all():
            possible_values.append(k)
            counts.append(value_mapping[k])

    probabilities = np.array(counts) / sum(counts)
    return random_sampler.choice(possible_values, p=probabilities)


def swap_value(system: System, unstable_coords: NDArray[np.int_]) -> System:
    """Perform a swap operation on the system at a random unstable coordinate.
    For a swap, a component is replaced with a new component from the system.
    Only unstable coordinates are considered for the swap operation to ensure convergence.

    Parameters
    ----------
    system : System
    unstable_coords : NDArray[np.int_]
        System Coordinates that are unstable and can be swapped.

    Returns
    -------
    System
        New System with updated state after swap operation.
    """
    idx = random_sampler.choice(len(unstable_coords))
    x, y = unstable_coords[idx]
    current_component = system.state[x, y]
    new_value = sample_swap_value(system, current_component)
    return system.perform_swap(x, y, new_value)


def translate_values(system: System, unstable_coords: NDArray[np.int_]) -> System:
    """Perform a translation operation on the system at a random unstable coordinate.
    For a translation, a component is moved to a new location in the system, the prior location is set to "empty".
    Only unstable coordinates are considered for the translation operation to ensure convergence.

    Parameters
    ----------
    system : System
    unstable_coords : NDArray[np.int_]
        System Coordinates that are unstable and can be swapped.

    Returns
    -------
    System
        New System with updated state after translation operation.
    """
    src_idx, dst_idx = random_sampler.choice(
        len(unstable_coords), size=2, replace=False
    )
    xa, ya = unstable_coords[src_idx]
    xb, yb = unstable_coords[dst_idx]
    return system.perform_translation(
        xa, ya, xb, yb, empty_state=system.default_component
    )
