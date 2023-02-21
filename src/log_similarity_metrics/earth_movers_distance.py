from collections import Counter
from typing import Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def earth_movers_distance(hist_1: list, hist_2: list, extra_mass: float = 1.0) -> float:
    """
    Compute the Earth Mover's Distance (EMD) between two histograms given the 1D array of observations. The EMD corresponds to the amount of
    observations that have to be moved (multiplied by the distance of the movement) to transform one histogram into the other. If one of the
    histograms has more observations than the other, each extra observation is penalized by [extra_mass].

    :param hist_1: 1D array with the observations of histogram 1.
    :param hist_2: 1D array with the observations of histogram 2.
    :param extra_mass: Penalization for extra observation.
    :return: The Earth Mover's Distance (EMD) between [hist_1] and [hist_2].
    """
    # Remove similar observations
    hist_1, hist_2 = _clean_histograms(hist_1, hist_2)
    if len(hist_1) == len(hist_2):
        # Sort
        hist_1.sort()
        hist_2.sort()
        # Compute distance as the sum of absolute movements element by element
        distance = 0
        for i in range(len(hist_1)):
            distance += abs(hist_1[i] - hist_2[i])
    else:
        # Swipe them if [hist_1] is smaller than [hist_2], so [hist_1] is always the larger histogram (or equal size)
        hist_1, hist_2 = (hist_2, hist_1) if len(hist_1) < len(hist_2) else (hist_1, hist_2)
        # Create cost matrix
        cost_matrix = np.zeros((len(hist_1), len(hist_1)))
        # Cost of movement of each observation in [hist_1] to each other observation in [hist_2]
        for i in range(len(hist_1)):
            for j in range(len(hist_2)):
                # Distance of moving observation [i] from [hist_1] to observation [j] from [hist_2]
                cost_matrix[i][j] = abs(hist_1[i] - hist_2[j])
            for j in range(len(hist_1) - len(hist_2)):
                # Distance of considering observation [i] from [hist_1] as extra observation (deletion)
                cost_matrix[i][j + len(hist_2)] = extra_mass
        # Compute combination of movements that minimize the total cost
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # Compute distance
        distance = cost_matrix[row_ind, col_ind].sum()
    # Return distance
    return distance


def _clean_histograms(hist_1: list, hist_2: list) -> Tuple[list, list]:
    """
    Remove from two histograms the observations that appear in both of them.

    :param hist_1: histogram 1.
    :param hist_2: histogram 2.

    :return: both histograms without the common observations.
    """
    # Get frequency of each observation
    hist_1_freq = Counter(hist_1)
    hist_2_freq = Counter(hist_2)
    # Create list without elements present in the other list (considering duplicates)
    clean_hist_1 = [
        element
        for i in hist_1_freq
        for element in [i] * max(hist_1_freq[i] - hist_2_freq.get(i, 0), 0)
    ]
    clean_hist_2 = [
        element
        for i in hist_2_freq
        for element in [i] * max(hist_2_freq[i] - hist_1_freq.get(i, 0), 0)
    ]
    # Return clean histograms
    return clean_hist_1, clean_hist_2
