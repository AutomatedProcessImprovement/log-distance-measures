from typing import Tuple

import pandas as pd
from scipy.stats import wasserstein_distance

from log_distance_measures.config import EventLogIDs, AbsoluteTimestampType, DistanceMetric, discretize_to_hour
from log_distance_measures.earth_movers_distance import earth_movers_distance


def absolute_event_distribution_distance(
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
    EMD (or Wasserstein Distance) between the distribution of timestamps of two event logs. To get this distribution, the timestamps are
    discretized to bins of size given by [discretize_event] (default by hour).

    :param original_log: first event log.
    :param original_ids: mapping for the column IDs of the first event log.
    :param simulated_log: second event log.
    :param simulated_ids: mapping for the column IDs for the second event log.
    :param discretize_type: type of EMD measure (only take into account start timestamps, only end timestamps, or both).
    :param discretize_event: function to discretize the total amount of seconds each timestamp represents, default to hour.
    :param metric: distance metric to use in the histogram comparison.
    :param normalize: whether to normalize the distance metric to a value in [0.0, 1.0].

    :return: the EMD between the timestamp distribution of the two event logs, measuring the amount of movements (considering their
    distance) to transform one timestamp histogram into the other.
    """
    # Get discretized start and/or end timestamps
    original_discrete_events, simulated_discrete_events = _discretize(
        original_log, original_ids, simulated_log, simulated_ids, discretize_type, discretize_event
    )
    # Compute distance metric
    if metric == DistanceMetric.EMD:
        distance = earth_movers_distance(original_discrete_events, simulated_discrete_events) / len(original_discrete_events)
    else:
        distance = wasserstein_distance(original_discrete_events, simulated_discrete_events)
    # Normalize if needed
    if normalize:
        print("WARNING! The normalization of a Wasserstein Distance is sensitive to the range of the two samples, "
              "long samples may cause a higher reduction of the error.")
        max_value = max(max(original_discrete_events), max(simulated_discrete_events))
        distance = distance / max_value if max_value > 0 else distance
    # Return metric
    return distance


def _discretize(
        original_log: pd.DataFrame,
        original_ids: EventLogIDs,
        simulated_log: pd.DataFrame,
        simulated_ids: EventLogIDs,
        discretize_type: AbsoluteTimestampType = AbsoluteTimestampType.BOTH,
        discretize_event=discretize_to_hour  # function to discretize a total amount of seconds into bins
) -> Tuple[list, list]:
    """
        Discretize the absolute timestamps (start, end, or both, depending on [discretize_type]) of the events in logs [event_log_1] and
        [event_log_2] using the function [discretize_event] (to absolute hours by default). When discretizing the timestamps, the first
        hour is always 0.

        :param original_log: first event log.
        :param original_ids: mapping for the column IDs of the first event log.
        :param simulated_log: second event log.
        :param simulated_ids: mapping for the column IDs for the second event log.
        :param discretize_type: type of EMD measure (only take into account start timestamps, only end timestamps, or both).
        :param discretize_event: function to discretize the total amount of seconds each timestamp represents, default to hour.

        :return: A pair of lists with the discretized timestamps of [event_log_1] and [event_log_2], respectively.
        """
    # Get the first and last dates of the log
    if discretize_type == AbsoluteTimestampType.END:
        # Consider only the 'end' times
        interval_start = min(
            original_log[original_ids.end_time].min(),
            simulated_log[simulated_ids.end_time].min()
        ).floor(freq='H')
    else:
        # Consider only 'start' or both
        interval_start = min(
            original_log[original_ids.start_time].min(),
            simulated_log[simulated_ids.start_time].min()
        ).floor(freq='H')
    # Discretize each event to its corresponding "bin"
    orig_discretized_events, sim_discretized_events = [], []
    if discretize_type != AbsoluteTimestampType.END:
        # Consider either 'start' or both, so add the discretized start times
        orig_discretized_events += [
            discretize_event(difference.total_seconds())
            for difference in (original_log[original_ids.start_time] - interval_start)
        ]
        sim_discretized_events += [
            discretize_event(difference.total_seconds())
            for difference in (simulated_log[simulated_ids.start_time] - interval_start)
        ]
    if discretize_type != AbsoluteTimestampType.START:
        # Consider either 'end' or both, so add the discretized end times
        orig_discretized_events += [
            discretize_event(difference.total_seconds())
            for difference in (original_log[original_ids.end_time] - interval_start)
        ]
        sim_discretized_events += [
            discretize_event(difference.total_seconds())
            for difference in (simulated_log[simulated_ids.end_time] - interval_start)
        ]
    # Return discretized timestamps
    return orig_discretized_events, sim_discretized_events
