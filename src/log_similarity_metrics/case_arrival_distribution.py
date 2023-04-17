import datetime
import math

import pandas as pd
from scipy.stats import wasserstein_distance

from log_similarity_metrics.config import EventLogIDs, DistanceMetric, discretize_to_hour
from log_similarity_metrics.earth_movers_distance import earth_movers_distance


def case_arrival_distribution_distance(
        original_log: pd.DataFrame,
        original_ids: EventLogIDs,
        simulated_log: pd.DataFrame,
        simulated_ids: EventLogIDs,
        discretize_event=discretize_to_hour,  # function to discretize a total amount of seconds into bins
        metric: DistanceMetric = DistanceMetric.WASSERSTEIN,
        normalize: bool = False
) -> float:
    """
    EMD (or Wasserstein Distance) between the distribution of case arrival of two event logs. To get this distribution, the timestamps are
    discretized to bins of size given by [discretize_instance] (default by hour).

    :param original_log: first event log.
    :param original_ids: mapping for the column IDs of the first event log.
    :param simulated_log: second event log.
    :param simulated_ids: mapping for the column IDs for the second event log.
    :param discretize_event: function to discretize the total amount of seconds each timestamp represents, default to hour.
    :param metric: distance metric to use in the histogram comparison.
    :param normalize: whether to normalize the distance metric to a value in [0.0, 1.0]

    :return: the EMD between the case arrival distribution of the two event logs, measuring the amount of movements (considering their
    distance) to transform one timestamp histogram into the other.
    """
    # Get arrival events of each log
    original_arrivals = _get_arrival_events(original_log, original_ids)
    simulated_arrivals = _get_arrival_events(simulated_log, simulated_ids)
    # Get the first arrival to normalize
    first_arrival = min(
        original_arrivals[original_ids.start_time].min(),
        simulated_arrivals[simulated_ids.start_time].min()
    ).floor(freq='H')
    # Discretize each event to its corresponding "bin"
    original_discrete_arrivals = [
        discretize_event(difference.total_seconds())
        for difference in (original_arrivals[original_ids.start_time] - first_arrival)
    ]
    simulated_discrete_arrivals = [
        discretize_event(difference.total_seconds())
        for difference in (simulated_arrivals[simulated_ids.start_time] - first_arrival)
    ]
    # Compute distance metric
    if metric == DistanceMetric.EMD:
        distance = earth_movers_distance(original_discrete_arrivals, simulated_discrete_arrivals) / len(original_discrete_arrivals)
    else:
        distance = wasserstein_distance(original_discrete_arrivals, simulated_discrete_arrivals)
    # Normalize if needed
    if normalize:
        print("WARNING! The normalization of a Wasserstein Distance is sensitive to the range of the two samples, "
              "long samples may cause a higher reduction of the error.")
        max_value = max(max(original_discrete_arrivals), max(simulated_discrete_arrivals))
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
        original_log: pd.DataFrame,
        original_ids: EventLogIDs,
        simulated_log: pd.DataFrame,
        simulated_ids: EventLogIDs,
        bin_size: datetime.timedelta,
        metric: DistanceMetric = DistanceMetric.WASSERSTEIN,
        normalize: bool = False
) -> float:
    """
    EMD (or Wasserstein Distance) between the distribution of inter-arrival times of two event logs. To get this distribution, the
    inter-arrival times are discretized to bins of size [bin_size].

    :param original_log: first event log.
    :param original_ids: mapping for the column IDs of the first event log.
    :param simulated_log: second event log.
    :param simulated_ids: mapping for the column IDs for the second event log.
    :param bin_size: time interval to define the bin size.
    :param metric: distance metric to use in the histogram comparison.
    :param normalize: whether to normalize the distance metric to a value in [0.0, 1.0]

    :return: the EMD between the inter-arrival time distribution of the two event logs, measuring the amount of movements (considering their
    distance) to transform one inter-arrival time histogram into the other.
    """
    # Get inter-arrival times
    original_inter_arrivals = _get_inter_arrival_times(original_log, original_ids)
    simulated_inter_arrivals = _get_inter_arrival_times(simulated_log, simulated_ids)
    # Discretize each event to its corresponding "bin"
    original_discrete_inter_arrivals = [
        math.floor(inter_arrival / bin_size)
        for inter_arrival in original_inter_arrivals
    ]
    simulated_discrete_inter_arrivals = [
        math.floor(inter_arrival / bin_size)
        for inter_arrival in simulated_inter_arrivals
    ]
    # Compute distance metric
    if metric == DistanceMetric.EMD:
        distance = earth_movers_distance(
            original_discrete_inter_arrivals,
            simulated_discrete_inter_arrivals
        ) / len(original_discrete_inter_arrivals)
    else:
        distance = wasserstein_distance(
            original_discrete_inter_arrivals,
            simulated_discrete_inter_arrivals
        )
    # Normalize if needed
    if normalize:
        print("WARNING! The normalization of a Wasserstein Distance is sensitive to the range of the two samples, "
              "long samples may cause a higher reduction of the error.")
        max_value = max(max(original_discrete_inter_arrivals), max(simulated_discrete_inter_arrivals))
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
