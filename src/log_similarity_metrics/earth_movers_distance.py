from collections import Counter
from typing import Tuple, Union

import numpy as np
import pulp as pulp


def earth_movers_distance(obs_1: Union[list, dict], obs_2: Union[list, dict], extra_mass: int = 1):
    """
    Compute the Earth Mover's Distance (EMD) between two histograms given the 1D array of observations. The EMD corresponds to the amount of
    observations that have to be moved (multiplied by the distance of the movement) to transform one histogram into the other. If one of the
    histograms has more observations than the other, each extra observation is penalized by [extra_mass].

    :param obs_1: list with a 1D array with the observations of histogram 1, or dict with each bin and the number of observations.
    :param obs_2: list with a 1D array with the observations of histogram 2, or dict with each bin and the number of observations.
    :param extra_mass: Penalization for extra observation.
    :return: The Earth Mover's Distance (EMD) between [hist_1] and [hist_2].
    """
    # Transform the 1-D histograms to 2-D histograms removing similar values
    hist_1, hist_2 = _clean_histograms(obs_1, obs_2)
    total_mass_1, total_mass_2 = sum(hist_1.values()), sum(hist_2.values())
    len_1, len_2 = len(hist_1), len(hist_2)
    if len_1 > 0 and len_2 > 0:
        # Create cost matrix
        c = np.zeros((len_1, len_2))
        for inx_i, key_i in enumerate(hist_1):
            for idx_j, key_j in enumerate(hist_2):
                c[inx_i][idx_j] = abs(key_i - key_j)
        # Create optimization problem
        model = pulp.LpProblem("EMD", pulp.LpMinimize)
        # Define variable
        x = pulp.LpVariable.dicts("x", [(i, j) for i in range(len_1) for j in range(len_2)], lowBound=0, cat="Continuous")
        # Constraint: the sum of movements for each bin in [hist_1] has to be lower or equal to its mass
        for inx_i, key_i in enumerate(hist_1):
            model += pulp.lpSum([x[(inx_i, j)] for j in range(len_2)]) <= hist_1[key_i]
        # Constraint: the sum of movements for each bin in [hist_2] has to be lower or equal to its mass
        for idx_j, key_j in enumerate(hist_2):
            model += pulp.lpSum([x[(i, idx_j)] for i in range(len_1)]) <= hist_2[key_j]
        # Constraint: the total mass moved has to be equal to the minimum of the masses
        model += pulp.lpSum([x[(i, j)] for j in range(len_2) for i in range(len_1)]) == min(total_mass_1, total_mass_2)
        # Constraint: minimize the total cost of the movements
        model += pulp.lpSum([c[i][j] * x[(i, j)] for i in range(len_1) for j in range(len_2)])
        # Solve problem
        pulp.LpSolverDefault.msg = 0
        model.solve()
        pulp.LpSolverDefault.msg = 1
        # Return mass + penalization for the extra mass that was not moved
        distance = pulp.value(model.objective) + abs(total_mass_1 - total_mass_2) * extra_mass
    else:
        # One of them has size 0, compute the total extra mass in the other
        distance = total_mass_1 * extra_mass + total_mass_2 * extra_mass
    # Return distance
    return distance


def _clean_histograms(obs_1: Union[list, dict], obs_2: Union[list, dict]) -> Tuple[dict, dict]:
    """
    Transform two histograms (either 1-D list of observations or dictionary with 2-D) to two 2-D histograms without the observations that
    they have in common.

    :param obs_1: list with the 1-D histogram 1, or dict with the 2-D histogram 1.
    :param obs_2: list with the 1-D histogram 2, or dict with the 2-D histogram 2.

    :return: both histograms in 2-D space, without the common observations.
    """
    # Transform to 2-D histograms
    hist_1 = Counter(obs_1) if type(obs_1) is list else obs_1
    hist_2 = Counter(obs_2) if type(obs_2) is list else obs_2
    # Parse [hist_1] subtracting the mass that is already in [clean_hist_2]
    clean_hist_1 = {}
    for i in hist_1:
        intersection_value = max(hist_1[i] - hist_2.get(i, 0), 0)
        if intersection_value > 0:
            clean_hist_1[i] = intersection_value
    # Parse [hist_2] subtracting the mass that is already in [clean_hist_1]
    clean_hist_2 = {}
    for i in hist_2:
        intersection_value = max(hist_2[i] - hist_1.get(i, 0), 0)
        if intersection_value > 0:
            clean_hist_2[i] = intersection_value
    # Return clean histograms
    return clean_hist_1, clean_hist_2
