import argparse
import datetime
import enum
import math
import multiprocessing
import os
import string
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from statistics import mean
from typing import Tuple, Union

import numpy as np
import pandas as pd
import pulp as pulp
from jellyfish import damerau_levenshtein_distance
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance, kstest


@dataclass
class EventLogIDs:
    case: str = 'case_id'
    activity: str = 'Activity'
    enabled_time: str = 'enabled_time'
    start_time: str = 'start_time'
    end_time: str = 'end_time'
    resource: str = 'Resource'


# You can change these values if the simulated logs have different IDs
log_1_ids = EventLogIDs()
log_2_ids = EventLogIDs(
    case='case_id',
    activity='activity',
    enabled_time='enabled_time',
    start_time='start_time',
    end_time='end_time',
    resource='resource'
)


class DistanceMetric(enum.Enum):
    EMD = 0
    WASSERSTEIN = 1
    KS = 2


##############################
# - EARTH MOVER'S DISTANCE - #
##############################


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


###################################
# - ABSOLUTE EVENT DISTRIBUTION - #
###################################

def absolute_event_distribution_distance(
        event_log_1: pd.DataFrame,
        event_log_2: pd.DataFrame,
        metric: DistanceMetric = DistanceMetric.EMD
) -> float | Tuple[float, float]:
    """
    EMD (or Wasserstein Distance) between the distribution of timestamps of two event logs. To get this distribution, the timestamps are
    discretized to bins of size given by [discretize_instant] (default by hour).

    :param event_log_1: first event log.
    :param event_log_2: second event log.
    :param metric: distance metric to use in the histogram comparison.

    :return: the EMD between the timestamp distribution of the two event logs, measuring the amount of movements (considering their
    distance) to transform one timestamp histogram into the other.
    """
    # Get discretized start and/or end timestamps
    discretized_instants_1, discretized_instants_2 = _discretize_for_absolute(event_log_1, event_log_2)
    # Compute distance metric
    if metric == DistanceMetric.EMD:
        distance = earth_movers_distance(discretized_instants_1, discretized_instants_2) / len(discretized_instants_1)
    elif metric == DistanceMetric.WASSERSTEIN:
        distance = wasserstein_distance(discretized_instants_1, discretized_instants_2)
    else:
        distance = kstest(discretized_instants_1, discretized_instants_2)
    # Return metric
    return distance


def _discretize_for_absolute(event_log_1: pd.DataFrame, event_log_2: pd.DataFrame) -> Tuple[list, list]:
    """
    Discretize the absolute timestamps (start, end, or both, depending on [discretize_type]) of the events in logs [event_log_1] and
    [event_log_2] using the function [discretize_instant] (to absolute hours by default). When discretizing the timestamps, the first
    hour is always 0.

    :param event_log_1: first event log.
    :param event_log_2: second event log.

    :return: A pair of lists with the discretized timestamps of [event_log_1] and [event_log_2], respectively.
    """
    # Get the first and last dates of the log
    interval_start = min(event_log_1[log_1_ids.start_time].min(), event_log_2[log_2_ids.start_time].min()).floor(freq='H')
    # Discretize each instant to its corresponding "bin"
    discretized_instants_1 = [
                                 math.floor(difference.total_seconds() / 3600) for difference in
                                 (event_log_1[log_1_ids.start_time] - interval_start)
                             ] + [
                                 math.floor(difference.total_seconds() / 3600) for difference in
                                 (event_log_1[log_1_ids.end_time] - interval_start)
                             ]
    discretized_instants_2 = [
                                 math.floor(difference.total_seconds() / 3600) for difference in
                                 (event_log_2[log_2_ids.start_time] - interval_start)
                             ] + [
                                 math.floor(difference.total_seconds() / 3600) for difference in
                                 (event_log_2[log_2_ids.end_time] - interval_start)
                             ]
    # Return discretized timestamps
    return discretized_instants_1, discretized_instants_2


####################################
# - CIRCADIAN EVENT DISTRIBUTION - #
####################################

def circadian_event_distribution_distance(
        event_log_1: pd.DataFrame,
        event_log_2: pd.DataFrame,
        metric: DistanceMetric = DistanceMetric.EMD
) -> float | Tuple[float, float]:
    """
    EMD (or Wasserstein Distance) between the distribution of timestamps of two event logs, windowed by weekday (e.g. the instants
    happening on all Mondays are compared together), and discretized to their hour.

    :param event_log_1: first event log.
    :param event_log_2: second event log.
    :param metric: distance metric to use in the histogram comparison.

    :return: the EMD between the timestamp distribution of the two event logs windowed by weekday.
    """
    # Get discretized start and/or end timestamps
    discretized_instants_1 = _discretize_for_circadian(event_log_1, log_1_ids)
    discretized_instants_2 = _discretize_for_circadian(event_log_2, log_2_ids)
    # Compute the distance between the instant in the event logs for each weekday
    distances = []
    for week_day in range(7):  # All week days
        window_1 = discretized_instants_1[discretized_instants_1['weekday'] == week_day]['hour'].tolist()
        window_2 = discretized_instants_2[discretized_instants_2['weekday'] == week_day]['hour'].tolist()
        if len(window_1) > 0 and len(window_2) > 0:
            # Both have observations in this weekday
            if metric == DistanceMetric.EMD:
                distances += [earth_movers_distance(window_1, window_2) / len(window_1)]
            elif metric == DistanceMetric.WASSERSTEIN:
                distances += [wasserstein_distance(window_1, window_2)]
            else:
                distances += [kstest(window_1, window_2)]
        elif len(window_1) == 0 and len(window_2) == 0:
            # Both have no observations in this weekday
            if metric == DistanceMetric.EMD:
                distances += [0.0]
            elif metric == DistanceMetric.WASSERSTEIN:
                distances += [0.0]
            else:
                distances += [(0.0, 1.0)]
        else:
            # Only one has observations in this weekday, penalize with max distance value
            if metric == DistanceMetric.EMD:
                distances += [(len(window_1) + len(window_2))]  # Number of observations
            elif metric == DistanceMetric.WASSERSTEIN:
                distances += [23.0]  # 23 is the maximum wasserstein value for two histograms with values between 0 and 23.
            else:
                distances += [(1.0, 0.0)]
    # Compute distance metric
    if metric == DistanceMetric.KS:
        distance = (mean([distance[0] for distance in distances]), mean([distance[1] for distance in distances]))
    else:
        distance = mean(distances)
    # Return metric
    return distance


def _discretize_for_circadian(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs
) -> pd.DataFrame:
    """
        Create a pd.Dataframe with the hour (0-23) of the timestamps of the events
        in log [event_log] in one column ('hour'), and the day of the week in another column ('weekday').

        :param event_log: event log to extract the instants of.
        :param log_ids: mapping for the column IDs of the event log.

        :return: A pd.Dataframe with the hour of each discretized instant in the one column, and the weekday in other.
        """
    # Get the instants to discretize
    to_discretize = pd.concat(
        [event_log[log_ids.start_time], event_log[log_ids.end_time]]
    ).reset_index(drop=True).to_frame(name='instant')
    # Compute their weekday
    to_discretize['weekday'] = to_discretize['instant'].apply(lambda instant: instant.day_of_week)
    to_discretize['hour'] = to_discretize['instant'].apply(lambda instant: instant.hour)
    to_discretize.drop(['instant'], axis=1, inplace=True)
    # Return discretized timestamps
    return to_discretize


##################
# - CYCLE TIME - #
##################

def cycle_time_distribution_distance(
        event_log_1: pd.DataFrame,
        event_log_2: pd.DataFrame,
        bin_size: datetime.timedelta,
        metric: DistanceMetric = DistanceMetric.EMD
) -> float | Tuple[float, float]:
    """
    EMD (or Wasserstein Distance) between the distribution of cycle times of two event logs. To get this distribution, the cycle times are
    discretized to bins of size [bin_size].

    :param event_log_1: first event log.
    :param event_log_2: second event log.
    :param bin_size: time interval to define the bin size.
    :param metric: distance metric to use in the histogram comparison.

    :return: the EMD between the cycle time distribution of the two event logs, measuring the amount of movements (considering their
    distance) to transform one cycle time histogram into the other.
    """
    # Get trace durations of each trace for the first log
    trace_durations_1 = []
    for case, events in event_log_1.groupby(log_1_ids.case):
        trace_durations_1 += [events[log_1_ids.end_time].max() - events[log_1_ids.start_time].min()]
    # Get trace durations of each trace for the second log
    trace_durations_2 = []
    for case, events in event_log_2.groupby(log_2_ids.case):
        trace_durations_2 += [events[log_2_ids.end_time].max() - events[log_2_ids.start_time].min()]
    # Discretize each instant to its corresponding "bin"
    min_duration = min(trace_durations_1 + trace_durations_2)
    discretized_durations_1 = [math.floor((trace_duration - min_duration) / bin_size) for trace_duration in trace_durations_1]
    discretized_durations_2 = [math.floor((trace_duration - min_duration) / bin_size) for trace_duration in trace_durations_2]
    # Compute distance metric
    if metric == DistanceMetric.EMD:
        distance = earth_movers_distance(discretized_durations_1, discretized_durations_2) / len(discretized_durations_1)
    elif metric == DistanceMetric.WASSERSTEIN:
        distance = wasserstein_distance(discretized_durations_1, discretized_durations_2)
    else:
        distance = kstest(discretized_durations_1, discretized_durations_2)
    # Return metric
    return distance


##########################################
# - CASE ARRIVAL DISTRIBUTION DISTANCE - #
##########################################

def case_arrival_distribution_distance(
        event_log_1: pd.DataFrame,
        event_log_2: pd.DataFrame,
        metric: DistanceMetric = DistanceMetric.EMD
) -> float | Tuple[float, float]:
    """
    EMD (or Wasserstein Distance) between the distribution of case arrival of two event logs. To get this distribution, the timestamps are
    discretized to bins of size given by [discretize_instance] (default by hour).

    :param event_log_1: first event log.
    :param event_log_2: second event log.
    :param metric: distance metric to use in the histogram comparison.

    :return: the EMD between the case arrival distribution of the two event logs, measuring the amount of movements (considering their
    distance) to transform one timestamp histogram into the other.
    """
    # Get arrival events of each log
    arrivals_1 = _get_arrival_events(event_log_1, log_1_ids)
    arrivals_2 = _get_arrival_events(event_log_2, log_2_ids)
    # Get the first arrival to normalize
    first_arrival = min(arrivals_1[log_1_ids.start_time].min(), arrivals_2[log_2_ids.start_time].min()).floor(freq='H')
    # Discretize each instant to its corresponding "bin"
    discretized_arrivals_1 = [
        math.floor(difference.total_seconds() / 3600) for difference in (arrivals_1[log_1_ids.start_time] - first_arrival)
    ]
    discretized_arrivals_2 = [
        math.floor(difference.total_seconds() / 3600) for difference in (arrivals_2[log_2_ids.start_time] - first_arrival)
    ]
    # Compute distance metric
    if metric == DistanceMetric.EMD:
        distance = earth_movers_distance(discretized_arrivals_1, discretized_arrivals_2) / len(discretized_arrivals_1)
    elif metric == DistanceMetric.WASSERSTEIN:
        distance = wasserstein_distance(discretized_arrivals_1, discretized_arrivals_2)
    else:
        distance = kstest(discretized_arrivals_1, discretized_arrivals_2)
    # Return metric
    return distance


def _get_arrival_events(event_log: pd.DataFrame, log_ids: EventLogIDs) -> pd.DataFrame:
    """
    Get the first event of each trace w.r.t. their start time.

    :param event_log: event log to get the arrival events from.
    :param log_ids: mapping for the column IDs of the event log.

    :return: a pd.DataFrame with the first event (w.r.t. their start time) of each process trace.
    """
    # Get index of first event per trace
    ids_to_retain = []
    for case_id, events in event_log.groupby(log_ids.case):
        ids_to_retain += [events[log_ids.start_time].idxmin()]
    # Return first event of each trace
    return event_log.loc[ids_to_retain]


#############################################
# - RELATIVE EVENTS DISTRIBUTION DISTANCE - #
#############################################

def relative_event_distribution_distance(
        event_log_1: pd.DataFrame,
        event_log_2: pd.DataFrame,
        metric: DistanceMetric = DistanceMetric.EMD
) -> float | Tuple[float, float]:
    """
    EMD (or Wasserstein Distance) between the distribution of timestamps of two event logs relative to the start of their trace. To get this
    distribution, the timestamps are first relativized w.r.t. the start of their case (e.g. the first timestamps of a trace would be
    transformed to 0); then, the relative timestamps are discretized to bins of size given by [discretize_instant] (default by hour).

    :param event_log_1: first event log.
    :param event_log_2: second event log.
    :param metric: distance metric to use in the histogram comparison.

    :return: the EMD between the relative timestamp distribution of the two event logs, measuring the amount of movements (considering their
    distance) to transform one timestamp histogram into the other.
    """
    # Get discretized start and/or end timestamps
    relative_1 = _relativize_and_discretize(event_log_1, log_1_ids)
    relative_2 = _relativize_and_discretize(event_log_2, log_2_ids)
    # Compute distance metric
    if metric == DistanceMetric.EMD:
        distance = earth_movers_distance(relative_1, relative_2) / len(relative_1)
    elif metric == DistanceMetric.WASSERSTEIN:
        distance = wasserstein_distance(relative_1, relative_2)
    else:
        distance = kstest(relative_1, relative_2)
    return distance


def _relativize_and_discretize(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs
) -> list:
    """
    Transform each timestamp (start, end, or both, depending on [discretize_type]) in the event log by making it relative w.r.t. the start
    time of its trace, and then discretize them using the function [discretize_instant] (to absolute hours by default).

    :param event_log: event log.
    :param log_ids: mapping for the column IDs of the event log.

    :return: A list with the relative and discretized timestamps of the log.
    """
    # Make relative w.r.t. the start of their case
    relative_starts, relative_ends = [], []
    for case_id, events in event_log.groupby(log_ids.case):
        case_start = events[log_ids.start_time].min()
        relative_starts += [instant - case_start for instant in events[log_ids.start_time]]
        relative_ends += [instant - case_start for instant in events[log_ids.end_time]]
    # Discretize each instant to its corresponding "bin"
    discretized_instants = [
                               math.floor(difference.total_seconds() / 3600) for difference in relative_starts
                           ] + [
                               math.floor(difference.total_seconds() / 3600) for difference in relative_ends
                           ]
    # Return discretized timestamps
    return discretized_instants


##############################
# - ACTIVE CASES OVER TIME - #
##############################

def work_in_progress_distance(
        event_log_1: pd.DataFrame,
        event_log_2: pd.DataFrame
) -> float:
    """
    EMD (or Wasserstein Distance) between the distribution of active cases over time. To get this distribution, the percentage of each
    window that is covered by active cases is computed. For example, given the window from 10am to 11am, if there are three cases active
    during the whole window (1 + 1 + 1), one case active half of the window (0.5), and two cases active a quarter of the window (0.25 +
    0.25), the active value for that hour is 4.

    :param event_log_1: first event log.
    :param event_log_2: second event log.
    :param metric: distance metric to use in the histogram comparison.

    :return: the EMD between the distribution of active cases over time of the two event logs, measuring the amount of movements
    (considering their distance) to transform one timestamp histogram into the other.
    """
    # Get timeline (reset to day in case daily frequency is used)
    start = min(event_log_1[log_1_ids.start_time].min(), event_log_2[log_2_ids.start_time].min()).floor(freq='H')
    end = max(event_log_1[log_1_ids.end_time].max(), event_log_2[log_2_ids.end_time].max()).ceil(freq='H')
    # Compute the active area of each bin
    wip_1 = _compute_work_in_progress(event_log_1, log_1_ids, start, end, pd.Timedelta(hours=1))
    wip_2 = _compute_work_in_progress(event_log_2, log_2_ids, start, end, pd.Timedelta(hours=1))
    # Compute SAE over the histograms
    distance = sum([
        abs(wip_1.get(key, 0) - wip_2.get(key, 0))
        for key
        in set(list(wip_1.keys()) + list(wip_2.keys()))
    ]) / (sum(wip_1.values()) + sum(wip_2.values()))
    # Return metric
    return distance


def _compute_work_in_progress(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        start: pd.Timestamp,
        end: pd.Timestamp,
        window_size: pd.Timedelta
) -> dict:
    """
    Compute, for each bin of [window_size] size within the interval from [start] to [end], the percentage of "area" where there was an
    active case. For example, given the window from 10am to 11am, if there are three cases active during the whole window (1 + 1 + 1),
    one case active half of the window (0.5), and two cases active a quarter of the window (0.25 + 0.25), the active value for that hour
    is 4.

    :param event_log: first event log.
    :param log_ids: mapping for the column IDs of the first event log.
    :param start: timestamp denoting the start of the interval to search in.
    :param end: timestamp denoting the end of the interval to search in.
    :param window_size: window to check the number of cases at the beginning of it.

    :return: a dict with the ID of each window and the work in progress in it.
    """
    # Transform event logs to cases
    cases = []
    for _case_id, events in event_log.groupby(log_ids.case):
        cases += [{'start': events[log_ids.start_time].min(), 'end': events[log_ids.end_time].max()}]
    cases = pd.DataFrame(cases)
    # Go over each bin computing the active area
    wip = {}
    for offset in range(math.ceil((end - start) / window_size)):
        current_window_start = start + window_size * offset
        current_window_end = current_window_start + window_size
        # Compute overlapping intervals (0s if no overlapping)
        within_window = (np.minimum(cases['end'], current_window_end) - np.maximum(cases['start'], current_window_start))
        # Sum positive ones (within the current window) and normalize area
        wip_value = sum(within_window[within_window > pd.Timedelta(0)], pd.Timedelta(0)) / window_size
        if wip_value > 0:
            wip[offset] = wip_value
    # Return WiP dict
    return wip


#######################
# - N-GRAM DISTANCE - #
#######################

def n_gram_distribution_distance(
        event_log_1: pd.DataFrame,
        event_log_2: pd.DataFrame,
        n: int = 3
) -> float:
    """
    Compute the distance between the frequency of n-grams in two event logs.

    :param event_log_1: first event log.
    :param event_log_2: second event log.
    :param n: size of the n-grams to build (e.g. n=3 will compare sequences of 3 activity instances like ABC and ABD).

    :return: the sum of absolute errors between the frequency distribution of n-grams in [event_log_1] and [event_log_2].
    """
    # Map of each activity to a number
    activity_labels = list(set(event_log_1[log_1_ids.activity].unique().tolist() + event_log_2[log_2_ids.activity].unique().tolist()))
    # Build n-grams histogram for each event log
    n_histogram_1 = _compute_n_grams(event_log_1, log_1_ids, activity_labels, n)
    n_histogram_2 = _compute_n_grams(event_log_2, log_2_ids, activity_labels, n)
    # Fill each histogram with a 0 for the n_grams missing from the other histogram
    frequencies_1, frequencies_2 = [], []
    for key in set(list(n_histogram_1.keys()) + list(n_histogram_2.keys())):
        frequencies_1 += [
            n_histogram_1[key] if key in n_histogram_1 else 0
        ]
        frequencies_2 += [
            n_histogram_2[key] if key in n_histogram_2 else 0
        ]
    # Compute distance metric
    distance = sum([abs(x - y) for (x, y) in zip(frequencies_1, frequencies_2)]) / (sum(frequencies_1) + sum(frequencies_2))
    # Return metric
    return distance


def _compute_n_grams(event_log: pd.DataFrame, log_ids: EventLogIDs, activity_labels: list, n: int = 3) -> dict:
    """
    Compute the n-grams of activities (directly-follows) of an event log.

    :param event_log: event log to analyze.
    :param log_ids: mapping for the column IDs of the event log.
    :param activity_labels: list with the unique activity labels to map them to n-grams.
    :param n: size of the n-grams to compute.

    :return: a dict with the n-grams as key, and their absolute frequency as value.
    """
    # Extend activity IDs with "None" (for start end of trace)
    activity_labels = [None] + activity_labels
    # Compute n-grams
    n_grams = {}
    for case_id, events in event_log.groupby(log_ids.case):
        # List with the IDs of each activity
        events = [0] * (n - 1) + [
            activity_labels.index(event) for event in events.sort_values([log_ids.start_time, log_ids.end_time])[log_ids.activity]
        ] + [0] * (n - 1)
        # Go over the IDs in a n-sized window
        for i in range(len(events) - n + 1):
            n_gram = ",".join([str(event) for event in events[i: i + n]])
            n_grams[n_gram] = n_grams[n_gram] + 1 if n_gram in n_grams else 1
    # Return n_grams and their frequency
    return n_grams


#################################
# - CONTROL-FLOW LOG DISTANCE - #
#################################

def control_flow_log_distance(
        event_log_1: pd.DataFrame,
        event_log_2: pd.DataFrame
) -> float:
    """
    Compute the Control-Flow Log Distance (see "Camargo M, Dumas M, González-Rojas O. 2021. Discovering generative models
    from event logs: data-driven simulation vs deep learning. PeerJ Computer Science 7:e577 https://doi.org/10.7717/peerj-cs.577"
    for a detailed description of a similarity version of the metric).

    :param event_log_1: first event log.
    :param event_log_2: second event log.

    :return: the Control-Flow Log Distance measure between [event_log_1] and [event_log_2].
    """
    # Transform the event log to a list of character sequences representing the traces
    sequences_1, sequences_2 = _event_logs_to_activity_sequences(event_log_1, event_log_2)
    # Calculate the DL distance between each pair of traces
    cost_matrix = _compute_distance_matrix(sequences_1, sequences_2)
    # Get the optimum pairing
    row_indexes, col_indexes = linear_sum_assignment(cost_matrix)
    # Compute the Control-Flow Log Distance
    cfld = mean([cost_matrix[i_1, i_2] for i_1, i_2 in zip(row_indexes, col_indexes)])
    # Return the mean of the distances between the traces
    return cfld


def _event_logs_to_activity_sequences(
        event_log_1: pd.DataFrame,
        event_log_2: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transform two event logs into [pd.DataFrame]s with the case IDs as index, and the sequence of activities (mapped to single characters)
    as value.

    :param event_log_1: first event log.
    :param log_1_ids: mapping for the column IDs of the first event log.
    :param event_log_2: second event log.
    :param log_2_ids: mapping for the column IDs for the second event log.

    :return: a tuple of two [pd.DataFrame]s with the case IDs as index and its corresponding activity sequence (mapped to single characters)
             as values.
    """
    # Check if there are enough characters to map all activities
    characters = string.printable
    activities = set(list(event_log_1[log_1_ids.activity].unique()) + list(event_log_2[log_2_ids.activity].unique()))
    if len(activities) > len(characters):
        raise RuntimeError(
            "Not enough characters ({}) to map all activities ({}) in Damerau Levenshtein distance calculation!".format(
                len(characters),
                len(activities)
            )
        )
    # Declare mapping
    mapping = {activity: characters[index] for index, activity in enumerate(activities)}
    # Calculate event logs
    processed_event_log_1 = _event_log_to_activity_sequence(event_log_1, log_1_ids, mapping)
    processed_event_log_2 = _event_log_to_activity_sequence(event_log_2, log_2_ids, mapping)
    # Return value
    return processed_event_log_1.reset_index(drop=True), processed_event_log_2.reset_index(drop=True)


def _event_log_to_activity_sequence(event_log: pd.DataFrame, log_ids: EventLogIDs, mapping: dict) -> pd.DataFrame:
    """
    Transform an event log into a [pd.DataFrame] with the case IDs as index, and the sequence of activities (mapped to single characters)
    as value.

    :param event_log: event log to transform.
    :param log_ids: mapping for the column IDs of the event log.
    :param mapping: mapping between activities and single characters.

    :return: a [pd.DataFrame] with the case IDs as index and its corresponding activity sequence (mapped to single characters) as values.
    """
    # Define lists to store each case
    case_ids = []
    activity_sequences = []
    # For each case, map the activities to character sequence
    for case_id, events in event_log.groupby(log_ids.case):
        case_ids += [case_id]
        sorted_events = events.sort_values([log_ids.end_time, log_ids.start_time])
        activity_sequences += [
            "".join([mapping[activity] for activity in sorted_events[log_ids.activity]])
        ]
    # Return DataFrame with the mapped activity sequences
    return pd.DataFrame(data={'sequence': activity_sequences}, index=case_ids)


def _compute_distance_matrix(sequences_1: pd.DataFrame, sequences_2: pd.DataFrame) -> np.ndarray:
    """
    Compute the matrix of (string edit) distances between all the sequences in [sequences_1] and [sequences_2].

    :param sequences_1: first list of sequences.
    :param sequences_2: second list of sequences.
    :return: matrix of (string edit) distances between each sequence in [sequences_1] and each sequence in [sequences_2].
    """
    # Compute distances
    num_cores = multiprocessing.cpu_count()
    num_cores = num_cores - 1 if num_cores > 2 else num_cores
    if num_cores > 1:
        # Parallel computation
        num_traces = len(sequences_1)
        traces_per_work = math.ceil(num_traces / num_cores)
        splits = [
            (i * traces_per_work, min(((i + 1) * traces_per_work), num_traces))
            for i in range(num_cores) if i * traces_per_work < num_traces
        ]
        # Run in parallel
        with ProcessPoolExecutor(max_workers=num_cores) as pool:
            args = [
                (sequences_1[start:end],
                 sequences_2,
                 start)
                for (start, end) in splits
            ]
            comparison_results = pool.map(_compare_traces, args)
        # Save results
        distances = [distance for distances in comparison_results for distance in distances]
    else:
        # Compute in one thread
        distances = _compare_traces((sequences_1, sequences_2, 0))
    # Create matrix
    distance_matrix = pd.DataFrame(distances).set_index(['i', 'j'])
    # Return numpy
    return distance_matrix.unstack().to_numpy()


def _compare_traces(args) -> list:
    # Parse args
    sequences_1, sequences_2, start_index = args
    # Compute distances
    distances = []
    for i, sequence_1 in sequences_1['sequence'].items():
        for j, sequence_2 in sequences_2['sequence'].items():
            distance = damerau_levenshtein_distance(sequence_1, sequence_2) / max(len(sequence_1), len(sequence_2))
            distances += [{'i': i + start_index, 'j': j, 'distance': distance}]
    # Return computed distances
    return distances


###################
# - MAIN SCRIPT - #
###################

if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser(description='Compute temporal distance measures of one or a set '
                                                 'of simulated event logs w.r.t. the original event log.')
    parser.add_argument("-cfld", action="store_true", help="Flag to compute also the CFLD metric.")
    parser.add_argument("original_log", help="Path to the original event log in CSV format.")
    parser.add_argument("simulated_log", help="Path to the simulated event log (or directory with a set of "
                                              "simulated logs) in CSV format.")
    args = parser.parse_args()
    # Parse simulated log/logs path
    if os.path.isdir(args.simulated_log):
        simulated_paths = [
            os.path.join(args.simulated_log, f)
            for f in os.listdir(args.simulated_log)
            if os.path.isfile(os.path.join(args.simulated_log, f))
        ]
    else:
        simulated_paths = [args.simulated_log]
    # Process original event log
    original_log = pd.read_csv(args.original_log)
    original_log[log_1_ids.start_time] = pd.to_datetime(original_log[log_1_ids.start_time], utc=True)
    original_log[log_1_ids.end_time] = pd.to_datetime(original_log[log_1_ids.end_time], utc=True)
    # Compute bin size
    bin_size = max(
        [events[log_1_ids.end_time].max() - events[log_1_ids.start_time].min()
         for _, events in original_log.groupby(log_1_ids.case)]
    ) / 1000  # 1.000 bins
    # Compute metrics for each simulated log
    with open("output.csv", 'a') as outfile:
        outfile.write("name,")
        outfile.write("bigram,bigram_runtime,trigram,trigram_runtime,")
        if args.cfld:
            outfile.write("cfld,cfld_runtime,")
        outfile.write("absolute_emd,absolute_emd_runtime,absolute_wass,absolute_wass_runtime,")
        outfile.write("case_arrival_emd,case_arrival_emd_runtime,case_arrival_wass,case_arrival_wass_runtime,")
        outfile.write("circadian_emd,circadian_emd_runtime,circadian_wass,circadian_wass_runtime,")
        outfile.write("relative_emd,relative_emd_runtime,relative_wass,relative_wass_runtime,")
        outfile.write("wip,wip_runtime,")
        outfile.write("cycle_time_wass,cycle_time_wass_runtime\n")
    for simulated_path in simulated_paths:
        # Read
        simulated_log = pd.read_csv(simulated_path)
        simulated_log[log_2_ids.start_time] = pd.to_datetime(simulated_log[log_2_ids.start_time], utc=True)
        simulated_log[log_2_ids.end_time] = pd.to_datetime(simulated_log[log_2_ids.end_time], utc=True)
        # Measures
        # NGRAM
        start = time.time()
        bigram = n_gram_distribution_distance(original_log, simulated_log, n=2)
        bigram_runtime = time.time() - start
        print("2-Gram Distribution Distance: {} s".format(bigram_runtime))
        start = time.time()
        trigram = n_gram_distribution_distance(original_log, simulated_log, n=3)
        trigram_runtime = time.time() - start
        print("3-Gram Distribution Distance: {} s".format(trigram_runtime))
        # CFLD
        if args.cfld:
            start = time.time()
            cfld = control_flow_log_distance(original_log, simulated_log)
            cfld_runtime = time.time() - start
            print("Control-Flow Log Distance: {} s".format(cfld_runtime))
        # ABSOLUTE
        start = time.time()
        absolute_events_emd = absolute_event_distribution_distance(original_log, simulated_log, DistanceMetric.EMD)
        absolute_events_emd_runtime = time.time() - start
        print("Absolute Event Distribution EMD: {} s".format(absolute_events_emd_runtime))
        start = time.time()
        absolute_events_wass = absolute_event_distribution_distance(original_log, simulated_log, DistanceMetric.WASSERSTEIN)
        absolute_events_wass_runtime = time.time() - start
        print("Absolute Event Distribution Wasserstein: {} s".format(absolute_events_wass_runtime))
        # ARRIVALS
        start = time.time()
        arrivals_emd = case_arrival_distribution_distance(original_log, simulated_log, DistanceMetric.EMD)
        arrivals_emd_runtime = time.time() - start
        print("Case Arrival Distribution EMD: {} s".format(arrivals_emd_runtime))
        start = time.time()
        arrivals_wass = case_arrival_distribution_distance(original_log, simulated_log, DistanceMetric.WASSERSTEIN)
        arrivals_wass_runtime = time.time() - start
        print("Case Arrival Distribution Wasserstein: {} s".format(arrivals_wass_runtime))
        # CIRCADIAN
        start = time.time()
        circadian_events_emd = circadian_event_distribution_distance(original_log, simulated_log, DistanceMetric.EMD)
        circadian_events_emd_runtime = time.time() - start
        print("Circadian Event Distribution EMD: {} s".format(circadian_events_emd_runtime))
        start = time.time()
        circadian_events_wass = circadian_event_distribution_distance(original_log, simulated_log, DistanceMetric.WASSERSTEIN)
        circadian_events_wass_runtime = time.time() - start
        print("Circadian Event Distribution Wasserstein: {} s".format(circadian_events_wass_runtime))
        # RELATIVE
        start = time.time()
        relative_events_emd = relative_event_distribution_distance(original_log, simulated_log, DistanceMetric.EMD)
        relative_events_emd_runtime = time.time() - start
        print("Relative Event Distribution EMD: {} s".format(relative_events_emd_runtime))
        start = time.time()
        relative_events_wass = relative_event_distribution_distance(original_log, simulated_log, DistanceMetric.WASSERSTEIN)
        relative_events_wass_runtime = time.time() - start
        print("Relative Event Distribution Wasserstein: {} s".format(relative_events_wass_runtime))
        # Active Cases Over Time (WiP)
        start = time.time()
        wip = work_in_progress_distance(original_log, simulated_log)
        wip_runtime = time.time() - start
        print("Active Cases Over Time (WiP): {} s".format(wip_runtime))
        # CYCLE TIME
        start = time.time()
        cycle_time_wass = cycle_time_distribution_distance(original_log, simulated_log, bin_size, DistanceMetric.WASSERSTEIN)
        cycle_time_wass_runtime = time.time() - start
        print("Cycle Time Distribution Distance: {} s".format(cycle_time_wass_runtime))
        # Print
        with open("output.csv", 'a') as outfile:
            outfile.write("{},".format(
                simulated_path
            ))
            outfile.write("{},{},{},{},".format(
                bigram,
                bigram_runtime,
                trigram,
                trigram_runtime
            ))
            if args.cfld:
                outfile.write("{},{},".format(
                    cfld,
                    cfld_runtime
                ))
            outfile.write("{},{},{},{},".format(
                absolute_events_emd,
                absolute_events_emd_runtime,
                absolute_events_wass,
                absolute_events_wass_runtime
            ))
            outfile.write("{},{},{},{},".format(
                arrivals_emd,
                arrivals_emd_runtime,
                arrivals_wass,
                arrivals_wass_runtime
            ))
            outfile.write("{},{},{},{},".format(
                circadian_events_emd,
                circadian_events_emd_runtime,
                circadian_events_wass,
                circadian_events_wass_runtime
            ))
            outfile.write("{},{},{},{},".format(
                relative_events_emd,
                relative_events_emd_runtime,
                relative_events_wass,
                relative_events_wass_runtime
            ))
            outfile.write("{},{},".format(
                wip,
                wip_runtime
            ))
            outfile.write("{},{}\n".format(
                cycle_time_wass,
                cycle_time_wass_runtime
            ))
