import datetime
import math

import pandas as pd
from scipy.stats import wasserstein_distance

from log_similarity_metrics.absolute_event_distribution import discretize_to_hour
from log_similarity_metrics.config import EventLogIDs, DistanceMetric
from log_similarity_metrics.earth_movers_distance import earth_movers_distance


def case_arrival_distribution_distance(
        event_log_1: pd.DataFrame,
        log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        log_2_ids: EventLogIDs,
        discretize_instant=discretize_to_hour,  # function to discretize a total amount of seconds into bins
        metric: DistanceMetric = DistanceMetric.WASSERSTEIN,
        normalize: bool = False
) -> float:
    """
    EMD (or Wasserstein Distance) between the distribution of case arrival of two event logs. To get this distribution, the timestamps are
    discretized to bins of size given by [discretize_instance] (default by hour).

    :param event_log_1: first event log.
    :param log_1_ids: mapping for the column IDs of the first event log.
    :param event_log_2: second event log.
    :param log_2_ids: mapping for the column IDs for the second event log.
    :param discretize_instant: function to discretize the total amount of seconds each timestamp represents, default to hour.
    :param metric: distance metric to use in the histogram comparison.
    :param normalize: whether to normalize the distance metric to a value in [0.0, 1.0]

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
        discretize_instant(difference.total_seconds()) for difference in (arrivals_1[log_1_ids.start_time] - first_arrival)
    ]
    discretized_arrivals_2 = [
        discretize_instant(difference.total_seconds()) for difference in (arrivals_2[log_2_ids.start_time] - first_arrival)
    ]
    # Compute distance metric
    if metric == DistanceMetric.EMD:
        distance = earth_movers_distance(discretized_arrivals_1, discretized_arrivals_2) / len(discretized_arrivals_1)
    else:
        distance = wasserstein_distance(discretized_arrivals_1, discretized_arrivals_2)
    # Normalize if needed
    if normalize:
        print("WARNING! The normalization of a Wasserstein Distance is sensitive to the range of the two samples, "
              "long samples may cause a higher reduction of the error.")
        max_value = max(max(discretized_arrivals_1), max(discretized_arrivals_2))
        distance = distance / max_value if max_value > 0 else 0
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


def inter_arrival_distribution_distance(
        event_log_1: pd.DataFrame,
        log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        log_2_ids: EventLogIDs,
        bin_size: datetime.timedelta,
        metric: DistanceMetric = DistanceMetric.WASSERSTEIN,
        normalize: bool = False
) -> float:
    """
    EMD (or Wasserstein Distance) between the distribution of inter-arrival times of two event logs. To get this distribution, the
    inter-arrival times are discretized to bins of size [bin_size].

    :param event_log_1: first event log.
    :param log_1_ids: mapping for the column IDs of the first event log.
    :param event_log_2: second event log.
    :param log_2_ids: mapping for the column IDs for the second event log.
    :param bin_size: time interval to define the bin size.
    :param metric: distance metric to use in the histogram comparison.
    :param normalize: whether to normalize the distance metric to a value in [0.0, 1.0]

    :return: the EMD between the inter-arrival time distribution of the two event logs, measuring the amount of movements (considering their
    distance) to transform one inter-arrival time histogram into the other.
    """
    # Get inter-arrival times
    inter_arrivals_1 = _get_inter_arrival_times(event_log_1, log_1_ids)
    inter_arrivals_2 = _get_inter_arrival_times(event_log_2, log_2_ids)
    # Discretize each instant to its corresponding "bin"
    discretized_inter_arrivals_1 = [math.floor(inter_arrival / bin_size) for inter_arrival in inter_arrivals_1]
    discretized_inter_arrivals_2 = [math.floor(inter_arrival / bin_size) for inter_arrival in inter_arrivals_2]
    # Compute distance metric
    if metric == DistanceMetric.EMD:
        distance = earth_movers_distance(discretized_inter_arrivals_1, discretized_inter_arrivals_2) / len(discretized_inter_arrivals_1)
    else:
        distance = wasserstein_distance(discretized_inter_arrivals_1, discretized_inter_arrivals_2)
    # Normalize if needed
    if normalize:
        print("WARNING! The normalization of a Wasserstein Distance is sensitive to the range of the two samples, "
              "long samples may cause a higher reduction of the error.")
        max_value = max(max(discretized_inter_arrivals_1), max(discretized_inter_arrivals_2))
        distance = distance / max_value if max_value > 0 else 0
    # Return metric
    return distance


def _get_inter_arrival_times(event_log: pd.DataFrame, log_ids: EventLogIDs) -> list:
    """
    Get the list of inter-arrival times in [event_log] (i.e. the intervals between each arrival and the next one).

    :param event_log: event log to get the inter-arrival times from.
    :param log_ids: mapping for the column IDs of the event log.

    :return: a list with the inter-arrival times in [event_log].
    """
    # Get absolute arrivals
    arrivals = []
    for case, events in event_log.groupby(log_ids.case):
        arrivals += [events[log_ids.start_time].min()]
    arrivals.sort()
    # Compute times between each arrival and the next one
    inter_arrivals = []
    last_arrival = None
    for arrival in arrivals:
        if last_arrival:
            inter_arrivals += [arrival - last_arrival]
        last_arrival = arrival
    # Return list with inter-arrivals
    return inter_arrivals
