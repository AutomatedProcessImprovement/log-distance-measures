import pandas as pd
from scipy.stats import wasserstein_distance

from log_similarity_metrics.absolute_event_distribution import discretize_to_hour
from log_similarity_metrics.config import EventLogIDs, AbsoluteTimestampType


def relative_event_distribution_distance(
        event_log_1: pd.DataFrame,
        log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        log_2_ids: EventLogIDs,
        discretize_type: AbsoluteTimestampType = AbsoluteTimestampType.BOTH,
        discretize_instant=discretize_to_hour  # function to discretize a total amount of seconds into bins
) -> float:
    """
    EMD (or Wasserstein Distance) between the distribution of timestamps of two event logs relative to the start of their trace. To get this
    distribution, the timestamps are first relativized w.r.t. the start of their case (e.g. the first timestamps of a trace would be
    transformed to 0); then, the relative timestamps are discretized to bins of size given by [discretize_instant] (default by hour).

    :param event_log_1: first event log.
    :param log_1_ids: mapping for the column IDs of the first event log.
    :param event_log_2: second event log.
    :param log_2_ids: mapping for the column IDs for the second event log.
    :param discretize_type: type of timestamps to consider (only take into account start timestamps, only end timestamps, or both).
    :param discretize_instant: function to discretize the total amount of seconds each timestamp represents, default to hour.

    :return: the EMD between the relative timestamp distribution of the two event logs, measuring the amount of movements (considering their
    distance) to transform one timestamp histogram into the other.
    """
    # Get discretized start and/or end timestamps
    relative_1 = _relativize_and_discretize(event_log_1, log_1_ids, discretize_type, discretize_instant)
    relative_2 = _relativize_and_discretize(event_log_2, log_2_ids, discretize_type, discretize_instant)
    # Return EMD metric
    return wasserstein_distance(relative_1, relative_2)


def _relativize_and_discretize(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        discretize_type: AbsoluteTimestampType = AbsoluteTimestampType.BOTH,
        discretize_instant=discretize_to_hour  # function to discretize a total amount of seconds into bins
) -> list:
    """
    Transform each timestamp (start, end, or both, depending on [discretize_type]) in the event log by making it relative w.r.t. the start
    time of its trace, and then discretize them using the function [discretize_instant] (to absolute hours by default).

    :param event_log: event log.
    :param log_ids: mapping for the column IDs of the event log.
    :param discretize_type: type of EMD measure (only take into account start timestamps, only end timestamps, or both).
    :param discretize_instant: function to discretize the total amount of seconds each timestamp represents, default to hour.

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
    discretized_instants = []
    if discretize_type != AbsoluteTimestampType.END:
        # Consider either 'start' or both, so add the discretized start times
        discretized_instants += [
            discretize_instant(difference.total_seconds()) for difference in relative_starts
        ]
    if discretize_type != AbsoluteTimestampType.START:
        # Consider either 'end' or both, so add the discretized end times
        discretized_instants += [
            discretize_instant(difference.total_seconds()) for difference in relative_ends
        ]
    # Return discretized timestamps
    return discretized_instants
