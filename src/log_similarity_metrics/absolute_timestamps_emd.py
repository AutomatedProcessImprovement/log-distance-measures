import math

import pandas as pd
from scipy.stats import wasserstein_distance

from log_similarity_metrics.config import EventLogIDs, AbsoluteHourEmdType


def discretize_to_minute(seconds: int):
    return math.floor(seconds / 60)


def discretize_to_hour(seconds: int):
    return math.floor(seconds / 3600)


def discretize_to_day(seconds: int):
    return math.floor(seconds / 3600 / 24)


def absolute_timestamps_emd(
        event_log_1: pd.DataFrame,
        log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        log_2_ids: EventLogIDs,
        emd_type: AbsoluteHourEmdType = AbsoluteHourEmdType.BOTH,
        discretize=discretize_to_hour  # function to discretize a total amount of seconds into bins
) -> float:
    """
    EMD (or Wasserstein Distance) between the distribution of timestamps of two event logs. To get this distribution, the timestamps are
    discretized to bins of size given by [discretize] (default by hour).

    :param event_log_1: first event log.
    :param log_1_ids: mapping for the column IDs of the first event log.
    :param event_log_2: second event log.
    :param log_2_ids: mapping for the column IDs for the second event log.
    :param emd_type: type of EMD measure (only take into account start timestamps, only end timestamps, or both).
    :param discretize: function to discretize the total amount of seconds each timestamp represents, default to hour.

    :return: the EMD between the timestamp distribution of the two event logs, measuring the amount of movements (considering their
    distance) to transform one timestamp histogram into the other.
    """
    # Get the first and last dates of the log
    if emd_type == AbsoluteHourEmdType.BOTH:
        interval_start = min(event_log_1[log_1_ids.start_time].min(), event_log_2[log_2_ids.start_time].min())
    elif emd_type == AbsoluteHourEmdType.START:
        interval_start = min(event_log_1[log_1_ids.start_time].min(), event_log_2[log_2_ids.start_time].min())
    else:
        interval_start = min(event_log_1[log_1_ids.end_time].min(), event_log_2[log_2_ids.end_time].min())
    interval_start = interval_start.floor(freq='H')
    # Discretize each instant to its corresponding "bin"
    discretized_instants_1 = []
    if emd_type != AbsoluteHourEmdType.END:
        discretized_instants_1 += [
            discretize(difference.total_seconds()) for difference in (event_log_1[log_1_ids.start_time] - interval_start)
        ]
    if emd_type != AbsoluteHourEmdType.START:
        discretized_instants_1 += [
            discretize(difference.total_seconds()) for difference in (event_log_1[log_1_ids.end_time] - interval_start)
        ]
    # Discretize each instant to its corresponding "bin"
    discretized_instants_2 = []
    if emd_type != AbsoluteHourEmdType.END:
        discretized_instants_2 += [
            discretize(difference.total_seconds()) for difference in (event_log_2[log_2_ids.start_time] - interval_start)
        ]
    if emd_type != AbsoluteHourEmdType.START:
        discretized_instants_2 += [
            discretize(difference.total_seconds()) for difference in (event_log_2[log_2_ids.end_time] - interval_start)
        ]
    # Return EMD metric
    return wasserstein_distance(discretized_instants_1, discretized_instants_2)
