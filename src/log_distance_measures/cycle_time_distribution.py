import datetime
import math

import pandas as pd
from scipy.stats import wasserstein_distance

from log_distance_measures.config import EventLogIDs, DistanceMetric
from log_distance_measures.earth_movers_distance import earth_movers_distance


def cycle_time_distribution_distance(
        original_log: pd.DataFrame,
        original_ids: EventLogIDs,
        simulated_log: pd.DataFrame,
        simulated_ids: EventLogIDs,
        bin_size: datetime.timedelta,
        metric: DistanceMetric = DistanceMetric.WASSERSTEIN,
        normalize: bool = False
) -> float:
    """
    EMD (or Wasserstein Distance) between the distribution of cycle times of two event logs. To get this distribution, the cycle times are
    discretized to bins of size [bin_size].

    :param original_log: first event log.
    :param original_ids: mapping for the column IDs of the first event log.
    :param simulated_log: second event log.
    :param simulated_ids: mapping for the column IDs for the second event log.
    :param bin_size: time interval to define the bin size.
    :param metric: distance metric to use in the histogram comparison.
    :param normalize: whether to normalize the distance metric to a value in [0.0, 1.0].

    :return: the EMD between the cycle time distribution of the two event logs, measuring the amount of movements (considering their
    distance) to transform one cycle time histogram into the other.
    """
    # Get trace durations of each trace for the first log
    original_cycle_times = []
    for case, events in original_log.groupby(original_ids.case):
        original_cycle_times += [events[original_ids.end_time].max() - events[original_ids.start_time].min()]
    # Get trace durations of each trace for the second log
    simulated_cycle_times = []
    for case, events in simulated_log.groupby(simulated_ids.case):
        simulated_cycle_times += [events[simulated_ids.end_time].max() - events[simulated_ids.start_time].min()]
    # Discretize each event to its corresponding "bin"
    min_duration = min(original_cycle_times + simulated_cycle_times)
    original_discrete_ct = [
        math.floor((trace_duration - min_duration) / bin_size)
        for trace_duration in original_cycle_times
    ]
    simulated_discrete_ct = [
        math.floor((trace_duration - min_duration) / bin_size)
        for trace_duration in simulated_cycle_times
    ]
    # Compute distance metric
    if metric == DistanceMetric.EMD:
        distance = earth_movers_distance(original_discrete_ct, simulated_discrete_ct) / len(original_discrete_ct)
    else:
        distance = wasserstein_distance(original_discrete_ct, simulated_discrete_ct)
    # Normalize if needed
    if normalize:
        print("WARNING! The normalization of a Wasserstein Distance is sensitive to the range of the two samples, "
              "long samples may cause a higher reduction of the error.")
        max_value = max(max(original_discrete_ct), max(simulated_discrete_ct))
        distance = distance / max_value if max_value > 0 else distance
    # Return metric
    return distance
