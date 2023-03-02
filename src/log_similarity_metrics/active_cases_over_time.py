import math

import pandas as pd
from scipy.stats import wasserstein_distance

from log_similarity_metrics.config import EventLogIDs


def active_cases_over_time_distance(
        event_log_1: pd.DataFrame,
        log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        log_2_ids: EventLogIDs,
        window_size: pd.Timedelta = pd.Timedelta(hours=1),
        normalize: bool = False
) -> float:
    """
    EMD (or Wasserstein Distance) between the distribution of active cases over time. To get this distribution, the number of active cases
    at the beginning of each window of size [window_size] (from the whole logs timespans) are computed.

    :param event_log_1: first event log.
    :param log_1_ids: mapping for the column IDs of the first event log.
    :param event_log_2: second event log.
    :param log_2_ids: mapping for the column IDs for the second event log.
    :param window_size: window to check the number of cases at the beginning of it.
    :param normalize: whether to normalize the distance metric to a value in [0.0, 1.0]

    :return: the EMD between the distribution of active cases over time of the two event logs, measuring the amount of movements
    (considering their distance) to transform one timestamp histogram into the other.
    """
    # Get timeline (reset to day in case daily frequency is used)
    start = min(event_log_1[log_1_ids.start_time].min(), event_log_2[log_2_ids.start_time].min()).floor(freq='24H')
    end = max(event_log_1[log_1_ids.end_time].max(), event_log_2[log_2_ids.end_time].max()).ceil(freq='24H')
    # Get the number of cases in each hour
    wip_1 = _active_cases_over_time(event_log_1, log_1_ids, start, end, window_size)
    wip_2 = _active_cases_over_time(event_log_2, log_2_ids, start, end, window_size)
    # Transform to 1D array
    wip_1 = [element for i in range(len(wip_1)) for element in [i] * wip_1[i]]
    wip_2 = [element for i in range(len(wip_2)) for element in [i] * wip_2[i]]
    # Compute distance metric
    if len(wip_1) > 0 and len(wip_2) > 0:
        distance = wasserstein_distance(wip_1, wip_2)
        if normalize:
            print("WARNING! The normalization of a Wasserstein Distance is sensitive to the range of the two samples, "
                  "long samples may cause a higher reduction of the error.")
            max_value = max(max(wip_1), max(wip_2))
            distance = distance / max_value if max_value > 0 else 0
    elif len(wip_1) == 0 and len(wip_2) == 0:
        distance = 0
    else:
        distance = (start - end) / window_size
    # Return metric
    return distance


def _active_cases_over_time(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        start: pd.Timestamp,
        end: pd.Timestamp,
        window_size: pd.Timedelta = pd.Timedelta(hours=1)
) -> list:
    """
    Compute the 2D array with the number of active cases at the beginning of each hour in [event_log], from [start] to [end]. Where the
    index of each number in the list is the hour w.r.t. [start], and the value the number of active cases.

    :param event_log: event log.
    :param log_ids: mapping for the column IDs of the event log.
    :param start: timestamp of the start of the time-series.
    :param end: timestamp of the end of the time-series.
    :param window_size: window to check the number of cases at the beginning of it.

    :return: a 2D array with the number of active cases at the beginning of each [window].
    """
    # Store the case starts/ends
    timestamps, types = [], []
    for _, case_events in event_log.groupby(log_ids.case):
        timestamps += [case_events[log_ids.start_time].min(), case_events[log_ids.end_time].max()]
        types += ["start", "end"]
    # Add an event per start of each window
    num_windows = math.ceil((end - start) / window_size) + 1
    timestamps += [start + (window_size * offset) for offset in range(num_windows)]
    types += ["reset"] * num_windows
    # Create sorted list of dicts
    events = pd.DataFrame(
        {'time': timestamps, 'type': types}
    ).sort_values(['time', 'type'], ascending=[True, False]).values.tolist()
    # Go over them start->end counting the number of active cases at the beginning of each window
    wip = []
    i, active = 0, 0
    while i < len(events):
        if events[i][1] == "start":
            # New case starting
            active += 1
        elif events[i][1] == "end":
            # Case ending
            active -= 1
        else:
            # New window, store active cases in this window
            wip += [active]
        # Continue with next event
        i += 1
    # Return the number of active cases over time
    return wip
