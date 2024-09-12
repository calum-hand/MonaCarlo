# For each Monte Carlo cycle, compute the unstable overlap between the current system and the reference system.
# Then attempt a CM move, accept the move if it improves the system (reduces delta_e), and repeat.

import os

from tqdm import trange

from mona_carlo.system import System
from mona_carlo.trial import swap_value, translate_values
from mona_carlo.metrics import compute_system_delta
from mona_carlo.datasets import load_mona
from mona_carlo.media import images_to_video, numpy_to_image
from mona_carlo.sampler import random_sampler

n_mc_cycles = 2_500
moves = [swap_value, translate_values]
move_probabilities = [0.5, 0.5]  # equal probability for all moves

reference_system = System(state=load_mona())
system = reference_system.scramble()
scores = [compute_system_delta(system, reference_system)]

# create directory for storing output
if not os.path.exists("images"):
    os.mkdir("images")

for idx in trange(n_mc_cycles):
    unstable_coords = system.compute_unstable_overlap(reference_system)
    n_unstable = len(unstable_coords)

    if n_unstable > 1:
        # can perform any move
        monte_carlo_move = random_sampler.choice(moves, p=move_probabilities)
    elif n_unstable == 1:
        # translation requires two unstable points, hence only allow swap
        monte_carlo_move = swap_value
    else:
        # no further improvement possible so exit early
        break

    proposed_system = monte_carlo_move(system, unstable_coords)
    proposed_system_score = compute_system_delta(proposed_system, reference_system)

    if proposed_system_score <= scores[-1]:
        system = proposed_system
        scores.append(proposed_system_score)

    # output image for progress validation
    numpy_to_image(f"images/mona_{idx}.png", reference_system.state, system.state)
images_to_video([f"images/mona_{i}.png" for i in range(idx)], file_name="mona_reconstruction.mp4")
