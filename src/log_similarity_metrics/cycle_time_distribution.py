import datetime
import math

import pandas as pd
from scipy.stats import wasserstein_distance

from log_similarity_metrics.config import EventLogIDs


def cycle_time_distribution_distance(
        event_log_1: pd.DataFrame,
        log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        log_2_ids: EventLogIDs,
        bin_size: datetime.timedelta,
        normalize: bool = True
) -> float:
    """
    EMD (or Wasserstein Distance) between the distribution of cycle times of two event logs. To get this distribution, the cycle times are
    discretized to bins of size [bin_size].

    :param event_log_1: first event log.
    :param log_1_ids: mapping for the column IDs of the first event log.
    :param event_log_2: second event log.
    :param log_2_ids: mapping for the column IDs for the second event log.
    :param bin_size: time interval to define the bin size.
    :param normalize: whether to normalize the distance metric to a value in [0.0, 1.0]

    :return: the EMD between the cycle time distribution of the two event logs, measuring the amount of movements (considering their
    distance) to transform one cycle time histogram into the other.
    """
    # Get trace durations of each trace for the first log
    trace_durations_1 = []
    for case, events in event_log_1.groupby([log_1_ids.case]):
        trace_durations_1 += [events[log_1_ids.end_time].max() - events[log_1_ids.start_time].min()]
    # Get trace durations of each trace for the second log
    trace_durations_2 = []
    for case, events in event_log_2.groupby([log_2_ids.case]):
        trace_durations_2 += [events[log_2_ids.end_time].max() - events[log_2_ids.start_time].min()]
    # Discretize each instant to its corresponding "bin"
    min_duration = min(trace_durations_1 + trace_durations_2)
    discretized_durations_1 = [math.floor((trace_duration - min_duration) / bin_size) for trace_duration in trace_durations_1]
    discretized_durations_2 = [math.floor((trace_duration - min_duration) / bin_size) for trace_duration in trace_durations_2]
    # Compute distance metric
    distance = wasserstein_distance(discretized_durations_1, discretized_durations_2)
    if normalize:
        max_value = max(max(discretized_durations_1), max(discretized_durations_2))
        distance = distance / max_value if max_value > 0 else 0
    # Return metric
    return distance
