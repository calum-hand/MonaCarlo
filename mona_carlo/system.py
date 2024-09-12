from typing import Dict
from typing_extensions import Self

import numpy as np
from numpy.typing import NDArray


class System:
    """Class to represent a system of components in a 2D grid."""

    def __init__(self, state: NDArray, components: Dict[NDArray, int]=None, default_component: NDArray=None):
        """
        Parameters
        ----------
        state : NDArray
            Numpy array representing the state of the system.
        components : Dict[NDArray, int], optional
            Dict of system components mapped to counts, by default None
            Calculated from state if not provided.
        default_component : NDArray, optional
            Most common component, by default None
            Calculated from components if not provided.
        """
        self.state = state
        self.components = (
            dict(zip(*np.unique(self.state, return_counts=True)))
            if components is None
            else components
        )
        self.default_component = (
            max(self.components.items(), key=lambda pair: pair[1])[0]
            if default_component is None
            else default_component
        )

    def perform_swap(self, x: int, y: int, swap_val: NDArray) -> Self:
        """Change component / value at (x,y) to provided value

        Parameters
        ----------
        x : int
            x_coordinate of system
        y : int
            y_coordinate of system
        swap_val : NDArray
            New value to be placed at (x,y)

        Returns
        -------
        Self
            New System with updated state
        """
        new_state = self.state.copy()
        new_state[x, y] = swap_val
        return System(new_state, self.components, self.default_component)

    def perform_translation(self, xa: int, ya: int, xb: int, yb: int, empty_state: NDArray) -> Self:
        """Move component at (xa, ya) to (xb, yb), then set (xa, ya) to "empty"

        Parameters
        ----------
        xa : int
            x_coordinate of source component
        ya : int
            y_coordinate of source component
        xb : int
            x_coordinate of destination component
        yb : int
            y_coordinate of destination component
        empty_state : NDArray
            Value to be placed at (xa, ya) indicating the space is now "empty" due to translation

        Returns
        -------
        Self
            New System with updated state
        """
        new_state = self.state.copy()
        new_state[xb, yb], new_state[xa, ya] = new_state[xa, ya], empty_state
        return System(new_state, self.components, self.default_component)

    def compute_unstable_overlap(self, system: Self) -> NDArray:
        """_summary_

        Parameters
        ----------
        system : Self
            Other system to compare against.
            Must be same dimensions as self.

        Returns
        -------
        NDArray
            2D of indices where self and system have different component values.
        """
        return np.argwhere(np.all(self.state != system.state, axis=-1))

    def scramble(self) -> Self:
        """Produce a new system with the same components, but in a random order.

        Returns
        -------
        Self
            Randomised system from original components.
        """
        state = np.random.permutation(self.state.copy().reshape(-1, 3))
        return System(
            state.reshape(self.state.shape), self.components, self.default_component
        )
