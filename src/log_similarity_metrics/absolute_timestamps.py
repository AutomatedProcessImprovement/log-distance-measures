import math
from typing import Tuple

import pandas as pd
from dtw import dtw
from scipy.stats import wasserstein_distance

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
        [event_log_2] using the function [discretize_instant] (to absolute hours by default).

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


def absolute_timestamps_emd(
        event_log_1: pd.DataFrame,
        log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        log_2_ids: EventLogIDs,
        discretize_type: AbsoluteTimestampType = AbsoluteTimestampType.BOTH,
        discretize_instant=discretize_to_hour  # function to discretize a total amount of seconds into bins
) -> float:
    """
    EMD (or Wasserstein Distance) between the distribution of timestamps of two event logs. To get this distribution, the timestamps are
    discretized to bins of size given by [discretize] (default by hour).

    :param event_log_1: first event log.
    :param log_1_ids: mapping for the column IDs of the first event log.
    :param event_log_2: second event log.
    :param log_2_ids: mapping for the column IDs for the second event log.
    :param discretize_type: type of EMD measure (only take into account start timestamps, only end timestamps, or both).
    :param discretize_instant: function to discretize the total amount of seconds each timestamp represents, default to hour.

    :return: the EMD between the timestamp distribution of the two event logs, measuring the amount of movements (considering their
    distance) to transform one timestamp histogram into the other.
    """
    # Get discretized start and/or end timestamps
    discretized_instants_1, discretized_instants_2 = _discretize(
        event_log_1, log_1_ids, event_log_2, log_2_ids, discretize_type, discretize_instant
    )
    # Return EMD metric
    return wasserstein_distance(discretized_instants_1, discretized_instants_2)


def absolute_timestamps_dtw(
        event_log_1: pd.DataFrame,
        log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        log_2_ids: EventLogIDs,
        discretize_type: AbsoluteTimestampType = AbsoluteTimestampType.BOTH,
        discretize_instant=discretize_to_hour  # function to discretize a total amount of seconds into bins
) -> float:
    """
    Dynamic Time Warping metric between the distribution of timestamps of two event logs. To get this distribution, the timestamps are
    discretized to bins of size given by [discretize] (default by hour).

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
    min_instant = min(discretized_instants_1 + discretized_instants_2)
    max_instant = max(discretized_instants_1 + discretized_instants_2)
    hist_1, hist_2 = [], []
    for i in range(min_instant, max_instant):
        hist_1 += [discretized_instants_1.count(i)]
        hist_2 += [discretized_instants_2.count(i)]
    # Return EMD metric
    return dtw(discretized_instants_1, discretized_instants_2, keep_internals=True).distance
