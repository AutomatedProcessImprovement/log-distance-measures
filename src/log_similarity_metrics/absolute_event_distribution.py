import math
from typing import Tuple

import pandas as pd
from dtw import dtw
from scipy.stats import wasserstein_distance, kstest

from log_similarity_metrics.config import EventLogIDs, AbsoluteTimestampType


def discretize_to_minute(seconds: int):
    return math.floor(seconds / 60)


def discretize_to_hour(seconds: int):
    return math.floor(seconds / 3600)


def discretize_to_day(seconds: int):
    return math.floor(seconds / 3600 / 24)


def _discretize(
        event_log_1: pd.DataFrame,
        log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        log_2_ids: EventLogIDs,
        discretize_type: AbsoluteTimestampType = AbsoluteTimestampType.BOTH,
        discretize_instant=discretize_to_hour  # function to discretize a total amount of seconds into bins
) -> Tuple[list, list]:
    """
        Discretize the absolute timestamps (start, end, or both, depending on [discretize_type]) of the events in logs [event_log_1] and
        [event_log_2] using the function [discretize_instant] (to absolute hours by default). When discretizing the timestamps, the first
        hour is always 0.

        :param event_log_1: first event log.
        :param log_1_ids: mapping for the column IDs of the first event log.
        :param event_log_2: second event log.
        :param log_2_ids: mapping for the column IDs for the second event log.
        :param discretize_type: type of EMD measure (only take into account start timestamps, only end timestamps, or both).
        :param discretize_instant: function to discretize the total amount of seconds each timestamp represents, default to hour.

        :return: A pair of lists with the discretized timestamps of [event_log_1] and [event_log_2], respectively.
        """
    # Get the first and last dates of the log
    if discretize_type == AbsoluteTimestampType.END:
        # Consider only the 'end' times
        interval_start = min(event_log_1[log_1_ids.end_time].min(), event_log_2[log_2_ids.end_time].min()).floor(freq='H')
    else:
        # Consider only 'start' or both
        interval_start = min(event_log_1[log_1_ids.start_time].min(), event_log_2[log_2_ids.start_time].min()).floor(freq='H')
    # Discretize each instant to its corresponding "bin"
    discretized_instants_1, discretized_instants_2 = [], []
    if discretize_type != AbsoluteTimestampType.END:
        # Consider either 'start' or both, so add the discretized start times
        discretized_instants_1 += [
            discretize_instant(difference.total_seconds()) for difference in (event_log_1[log_1_ids.start_time] - interval_start)
        ]
        discretized_instants_2 += [
            discretize_instant(difference.total_seconds()) for difference in (event_log_2[log_2_ids.start_time] - interval_start)
        ]
    if discretize_type != AbsoluteTimestampType.START:
        # Consider either 'end' or both, so add the discretized end times
        discretized_instants_1 += [
            discretize_instant(difference.total_seconds()) for difference in (event_log_1[log_1_ids.end_time] - interval_start)
        ]
        discretized_instants_2 += [
            discretize_instant(difference.total_seconds()) for difference in (event_log_2[log_2_ids.end_time] - interval_start)
        ]
    # Return discretized timestamps
    return discretized_instants_1, discretized_instants_2


def absolute_event_distribution_distance(
        event_log_1: pd.DataFrame,
        log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        log_2_ids: EventLogIDs,
        discretize_type: AbsoluteTimestampType = AbsoluteTimestampType.BOTH,
        discretize_instant=discretize_to_hour,  # function to discretize a total amount of seconds into bins
        normalize: bool = False
) -> float:
    """
    EMD (or Wasserstein Distance) between the distribution of timestamps of two event logs. To get this distribution, the timestamps are
    discretized to bins of size given by [discretize_instant] (default by hour).

    :param event_log_1: first event log.
    :param log_1_ids: mapping for the column IDs of the first event log.
    :param event_log_2: second event log.
    :param log_2_ids: mapping for the column IDs for the second event log.
    :param discretize_type: type of EMD measure (only take into account start timestamps, only end timestamps, or both).
    :param discretize_instant: function to discretize the total amount of seconds each timestamp represents, default to hour.
    :param normalize: whether to normalize the distance metric to a value in [0.0, 1.0].

    :return: the EMD between the timestamp distribution of the two event logs, measuring the amount of movements (considering their
    distance) to transform one timestamp histogram into the other.
    """
    # Get discretized start and/or end timestamps
    discretized_instants_1, discretized_instants_2 = _discretize(
        event_log_1, log_1_ids, event_log_2, log_2_ids, discretize_type, discretize_instant
    )
    # Compute distance metric
    distance = wasserstein_distance(discretized_instants_1, discretized_instants_2)
    if normalize:
        print("WARNING! The normalization of a Wasserstein Distance is sensitive to the range of the two samples, "
              "long samples may cause a higher reduction of the error.")
        max_value = max(max(discretized_instants_1), max(discretized_instants_2))
        distance = distance / max_value if max_value > 0 else 0
    # Return metric
    return distance


def absolute_event_distribution_ks(
        event_log_1: pd.DataFrame,
        log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        log_2_ids: EventLogIDs,
        discretize_type: AbsoluteTimestampType = AbsoluteTimestampType.BOTH,
        discretize_instant=discretize_to_hour  # function to discretize a total amount of seconds into bins
) -> float:
    """
    Kolmogorov-Smirnov test between the distribution of timestamps of two event logs. To get this distribution, the timestamps are
    discretized to bins of size given by [discretize_instant] (default by hour).

    :param event_log_1: first event log.
    :param log_1_ids: mapping for the column IDs of the first event log.
    :param event_log_2: second event log.
    :param log_2_ids: mapping for the column IDs for the second event log.
    :param discretize_type: type of KS test (only take into account start timestamps, only end timestamps, or both).
    :param discretize_instant: function to discretize the total amount of seconds each timestamp represents, default to hour.

    :return: the Kolmogorov-Smirnov test statistic between the timestamp distribution of the two event logs.
    """
    # Get discretized start and/or end timestamps
    discretized_instants_1, discretized_instants_2 = _discretize(
        event_log_1, log_1_ids, event_log_2, log_2_ids, discretize_type, discretize_instant
    )
    # Return KS statistic
    return kstest(discretized_instants_1, discretized_instants_2)[0]  # Return the statistic of the Kolmogorov-Smirnov


def absolute_event_distribution_dtw(
        event_log_1: pd.DataFrame,
        log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        log_2_ids: EventLogIDs,
        discretize_type: AbsoluteTimestampType = AbsoluteTimestampType.BOTH,
        discretize_instant=discretize_to_hour  # function to discretize a total amount of seconds into bins
) -> float:
    """
    Dynamic Time Warping metric between the distribution of timestamps of two event logs. To get this distribution, the timestamps are
    discretized to bins of size given by [discretize_instant] (default by hour).

    :param event_log_1: first event log.
    :param log_1_ids: mapping for the column IDs of the first event log.
    :param event_log_2: second event log.
    :param log_2_ids: mapping for the column IDs for the second event log.
    :param discretize_type: type of EMD measure (only take into account start timestamps, only end timestamps, or both).
    :param discretize_instant: function to discretize the total amount of seconds each timestamp represents, default to hour.

    :return: the DTW metric between the timestamp distribution of the two event logs.
    """
    # Get discretized start and/or end timestamps
    discretized_instants_1, discretized_instants_2 = _discretize(
        event_log_1, log_1_ids, event_log_2, log_2_ids, discretize_type, discretize_instant
    )
    # Group them to build the histogram
    max_instant = max(max(discretized_instants_1), max(discretized_instants_2))
    hist_1 = [0] * (max_instant + 1)
    for i in discretized_instants_1:
        hist_1[i] += 1
    hist_2 = [0] * (max_instant + 1)
    for i in discretized_instants_2:
        hist_2[i] += 1
    # Add one 0 in the start and end of each histogram, to avoid penalization in case of late start/end in one of the cases
    hist_1 = [0] + hist_1 + [0]
    hist_2 = [0] + hist_2 + [0]
    # Return EMD metric
    return dtw(hist_1, hist_2, keep_internals=True).distance
