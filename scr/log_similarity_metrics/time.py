import datetime
import enum
import math

import pandas as pd
from scipy.stats import wasserstein_distance

from .config import EventLogIDs


class _EmdType(enum.Enum):
    BOTH = 0
    START = 1
    END = 2


def discretize_to_hour(seconds: int):
    return math.floor(seconds / 3600)


def discretize_to_day(seconds: int):
    return math.floor(seconds / 3600 / 24)


def absolute_hour_emd(
        event_log_1: pd.DataFrame,
        log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        log_2_ids: EventLogIDs,
        emd_type: _EmdType = _EmdType.BOTH,
        discretize=discretize_to_hour  # function to discretize a total amount of seconds
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
    if emd_type == _EmdType.BOTH:
        interval_start = min(event_log_1[log_1_ids.start_time].min(), event_log_2[log_2_ids.start_time].min())
    elif emd_type == _EmdType.START:
        interval_start = min(event_log_1[log_1_ids.start_time].min(), event_log_2[log_2_ids.start_time].min())
    else:
        interval_start = min(event_log_1[log_1_ids.end_time].min(), event_log_2[log_2_ids.end_time].min())
    interval_start = interval_start.floor(freq='H')
    # Discretize each instant to its corresponding "bin"
    discretized_instants_1 = []
    if emd_type != _EmdType.END:
        discretized_instants_1 += [
            discretize(difference.total_seconds()) for difference in (event_log_1[log_1_ids.start_time] - interval_start)
        ]
    if emd_type != _EmdType.START:
        discretized_instants_1 += [
            discretize(difference.total_seconds()) for difference in (event_log_1[log_1_ids.end_time] - interval_start)
        ]
    # Discretize each instant to its corresponding "bin"
    discretized_instants_2 = []
    if emd_type != _EmdType.END:
        discretized_instants_2 += [
            discretize(difference.total_seconds()) for difference in (event_log_2[log_2_ids.start_time] - interval_start)
        ]
    if emd_type != _EmdType.START:
        discretized_instants_2 += [
            discretize(difference.total_seconds()) for difference in (event_log_2[log_2_ids.end_time] - interval_start)
        ]
    # Return EMD metric
    return wasserstein_distance(discretized_instants_1, discretized_instants_2)


def cycle_time_emd(
        event_log_1: pd.DataFrame,
        log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        log_2_ids: EventLogIDs,
        bin_size: datetime.timedelta
) -> float:
    """
    EMD (or Wasserstein Distance) between the distribution of cycle times of two event logs. To get this distribution, the cycle times are
    discretized to bins of size [bin_size].

    :param event_log_1: first event log.
    :param log_1_ids: mapping for the column IDs of the first event log.
    :param event_log_2: second event log.
    :param log_2_ids: mapping for the column IDs for the second event log.
    :param bin_size: time interval to define the bin size.

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
    # Return EMD metric
    return wasserstein_distance(discretized_durations_1, discretized_durations_2)
