import math

import numpy as np
import pandas as pd

from log_similarity_metrics.config import EventLogIDs


def work_in_progress_distance(
        original_log: pd.DataFrame,
        original_ids: EventLogIDs,
        simulated_log: pd.DataFrame,
        simulated_ids: EventLogIDs,
        window_size: pd.Timedelta = pd.Timedelta(hours=1),
        normalize: bool = True
) -> float:
    """
    EMD (or Wasserstein Distance) between the distribution of active cases over time. To get this distribution, the percentage of each
    window that is covered by active cases is computed. For example, given the window from 10am to 11am, if there are three cases active
    during the whole window (1 + 1 + 1), one case active half of the window (0.5), and two cases active a quarter of the window (0.25 +
    0.25), the active value for that hour is 4.

    :param original_log: first event log.
    :param original_ids: mapping for the column IDs of the first event log.
    :param simulated_log: second event log.
    :param simulated_ids: mapping for the column IDs for the second event log.
    :param window_size: window to check the number of cases at the beginning of it.
    :param normalize: whether to normalize the distance metric to a value in [0.0, 1.0]

    :return: the EMD between the distribution of active cases over time of the two event logs, measuring the amount of movements
    (considering their distance) to transform one timestamp histogram into the other.
    """
    # Get timeline (reset to day in case daily frequency is used)
    start = min(
        original_log[original_ids.start_time].min(),
        simulated_log[simulated_ids.start_time].min()
    ).floor(freq='24H')
    end = max(
        original_log[original_ids.end_time].max(),
        simulated_log[simulated_ids.end_time].max()
    ).ceil(freq='24H')
    # Compute the active area of each bin
    original_wip = _compute_work_in_progress(original_log, original_ids, start, end, window_size)
    simulated_wip = _compute_work_in_progress(simulated_log, simulated_ids, start, end, window_size)
    # Compute SAE over the histograms
    distance = sum([
        abs(original_wip.get(key, 0) - simulated_wip.get(key, 0))
        for key
        in set(list(original_wip.keys()) + list(simulated_wip.keys()))
    ])
    if normalize:
        distance = distance / (sum(original_wip.values()) + sum(simulated_wip.values()))
    # Return metric
    return distance


def _compute_work_in_progress(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
        start: pd.Timestamp,
        end: pd.Timestamp,
        window_size: pd.Timedelta
) -> dict:
    """
    Compute, for each bin of [window_size] size within the interval from [start] to [end], the percentage of "area" where there was an
    active case. For example, given the window from 10am to 11am, if there are three cases active during the whole window (1 + 1 + 1),
    one case active half of the window (0.5), and two cases active a quarter of the window (0.25 + 0.25), the active value for that hour
    is 4.

    :param event_log: first event log.
    :param log_ids: mapping for the column IDs of the first event log.
    :param start: timestamp denoting the start of the interval to search in.
    :param end: timestamp denoting the end of the interval to search in.
    :param window_size: window to check the number of cases at the beginning of it.

    :return: a dict with the ID of each window and the work in progress in it.
    """
    # Transform event logs to cases
    cases = []
    for _case_id, events in event_log.groupby(log_ids.case):
        cases += [{'start': events[log_ids.start_time].min(), 'end': events[log_ids.end_time].max()}]
    cases = pd.DataFrame(cases)
    # Go over each bin computing the active area
    wip = {}
    for offset in range(math.ceil((end - start) / window_size)):
        current_window_start = start + window_size * offset
        current_window_end = current_window_start + window_size
        # Compute overlapping intervals (0s if no overlapping)
        within_window = (np.minimum(cases['end'], current_window_end) - np.maximum(cases['start'], current_window_start))
        # Sum positive ones (within the current window) and normalize area
        wip_value = sum(within_window[within_window > pd.Timedelta(0)], pd.Timedelta(0)) / window_size
        if wip_value > 0:
            wip[offset] = wip_value
    # Return WiP dict
    return wip


def num_active_cases_over_time_distance(
        original_log: pd.DataFrame,
        original_ids: EventLogIDs,
        simulated_log: pd.DataFrame,
        simulated_ids: EventLogIDs,
        window_size: pd.Timedelta = pd.Timedelta(hours=1),
        normalize: bool = True
) -> float:
    """
    EMD (or Wasserstein Distance) between the distribution of active cases over time. To get this distribution, the number of active cases
    at the beginning of each window of size [window_size] (from the whole logs timespans) are computed.

    :param original_log: first event log.
    :param original_ids: mapping for the column IDs of the first event log.
    :param simulated_log: second event log.
    :param simulated_ids: mapping for the column IDs for the second event log.
    :param window_size: window to check the number of cases at the beginning of it.
    :param normalize: whether to normalize the distance metric to a value in [0.0, 1.0]

    :return: the EMD between the distribution of active cases over time of the two event logs, measuring the amount of movements
    (considering their distance) to transform one timestamp histogram into the other.
    """
    # Get timeline (reset to day in case daily frequency is used)
    start = min(
        original_log[original_ids.start_time].min(),
        simulated_log[simulated_ids.start_time].min()
    ).floor(freq='24H')
    end = max(
        original_log[original_ids.end_time].max(),
        simulated_log[simulated_ids.end_time].max()
    ).ceil(freq='24H')
    # Get the number of cases in each hour
    original_acot = _num_active_cases_over_time(original_log, original_ids, start, end, window_size)
    simulated_acot = _num_active_cases_over_time(simulated_log, simulated_ids, start, end, window_size)
    # Compute SAE over the histograms
    distance = sum([
        abs(original_acot[i] - simulated_acot[i])
        for i
        in range(max(len(original_acot), len(simulated_acot)))
    ])
    if normalize:
        distance = distance / (sum(original_acot) + sum(simulated_acot))
    # Return metric
    return distance


def _num_active_cases_over_time(
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
    timestamps += [start + (window_size * offset) for offset in range(num_windows + 1)]
    types += ["window"] * (num_windows + 1)
    # Create sorted list of dicts
    events = pd.DataFrame(
        {'time': timestamps, 'type': types}
    ).sort_values(['time', 'type'], ascending=[True, False]).values.tolist()
    # Go over them start->end counting the number of active cases at the beginning of each window
    active_cases_over_time = []
    i, active, active_current_window = 0, 0, 0
    while i < len(events):
        if events[i][1] == "start":
            # New case starting
            active += 1
            active_current_window += 1
        elif events[i][1] == "end":
            # Case ending
            active -= 1
        else:
            # New window, store active cases in this window
            active_cases_over_time += [active_current_window]
            active_current_window = active
        # Continue with next event
        i += 1
    # Return the number of active cases over time
    return active_cases_over_time
