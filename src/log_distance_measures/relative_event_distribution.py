import pandas as pd
from scipy.stats import wasserstein_distance

from log_distance_measures.config import EventLogIDs, AbsoluteTimestampType, DistanceMetric, discretize_to_hour
from log_distance_measures.earth_movers_distance import earth_movers_distance


def relative_event_distribution_distance(
        original_log: pd.DataFrame,
        original_ids: EventLogIDs,
        simulated_log: pd.DataFrame,
        simulated_ids: EventLogIDs,
        discretize_type: AbsoluteTimestampType = AbsoluteTimestampType.BOTH,
        discretize_event=discretize_to_hour,  # function to discretize a total amount of seconds into bins
        metric: DistanceMetric = DistanceMetric.WASSERSTEIN,
        normalize: bool = False
) -> float:
    """
    EMD (or Wasserstein Distance) between the distribution of timestamps of two event logs relative to the start of their trace. To get this
    distribution, the timestamps are first relativized w.r.t. the start of their case (e.g. the first timestamps of a trace would be
    transformed to 0); then, the relative timestamps are discretized to bins of size given by [discretize_event] (default by hour).

    :param original_log: first event log.
    :param original_ids: mapping for the column IDs of the first event log.
    :param simulated_log: second event log.
    :param simulated_ids: mapping for the column IDs for the second event log.
    :param discretize_type: type of timestamps to consider (only take into account start timestamps, only end timestamps, or both).
    :param discretize_event: function to discretize the total amount of seconds each timestamp represents, default to hour.
    :param metric: distance metric to use in the histogram comparison.
    :param normalize: whether to normalize the distance metric to a value in [0.0, 1.0]

    :return: the EMD between the relative timestamp distribution of the two event logs, measuring the amount of movements (considering their
    distance) to transform one timestamp histogram into the other.
    """
    # Get discretized start and/or end timestamps
    relative_1 = _relativize_and_discretize(original_log, original_ids, discretize_type, discretize_event)
    relative_2 = _relativize_and_discretize(simulated_log, simulated_ids, discretize_type, discretize_event)
    # Compute distance metric
    if metric == DistanceMetric.EMD:
        distance = earth_movers_distance(relative_1, relative_2) / len(relative_1)
    else:
        distance = wasserstein_distance(relative_1, relative_2)
    # Normalize if needed
    if normalize:
        print("WARNING! The normalization of a Wasserstein Distance is sensitive to the range of the two samples, "
              "long samples may cause a higher reduction of the error.")
        max_value = max(max(relative_1), max(relative_2))
        distance = distance / max_value if max_value > 0 else distance
    return distance


def _relativize_and_discretize(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        discretize_type: AbsoluteTimestampType = AbsoluteTimestampType.BOTH,
        discretize_event=discretize_to_hour  # function to discretize a total amount of seconds into bins
) -> list:
    """
    Transform each timestamp (start, end, or both, depending on [discretize_type]) in the event log by making it relative w.r.t. the start
    time of its trace, and then discretize them using the function [discretize_event] (to absolute hours by default).

    :param event_log: event log.
    :param log_ids: mapping for the column IDs of the event log.
    :param discretize_type: type of EMD measure (only take into account start timestamps, only end timestamps, or both).
    :param discretize_event: function to discretize the total amount of seconds each timestamp represents, default to hour.

    :return: A list with the relative and discretized timestamps of the log.
    """
    # Make relative w.r.t. the start of their case
    relative_starts, relative_ends = [], []
    for case_id, events in event_log.groupby(log_ids.case):
        case_start = events[log_ids.start_time].min()
        if discretize_type != AbsoluteTimestampType.END:
            # Consider either 'start' or both, so add the discretized start times
            relative_starts += [instant - case_start for instant in events[log_ids.start_time]]
        if discretize_type != AbsoluteTimestampType.START:
            # Consider either 'end' or both, so add the discretized end times
            relative_ends += [instant - case_start for instant in events[log_ids.end_time]]
    # Discretize each instant to its corresponding "bin"
    discretized_events = []
    if discretize_type != AbsoluteTimestampType.END:
        # Consider either 'start' or both, so add the discretized start times
        discretized_events += [
            discretize_event(difference.total_seconds()) for difference in relative_starts
        ]
    if discretize_type != AbsoluteTimestampType.START:
        # Consider either 'end' or both, so add the discretized end times
        discretized_events += [
            discretize_event(difference.total_seconds()) for difference in relative_ends
        ]
    # Return discretized timestamps
    return discretized_events
