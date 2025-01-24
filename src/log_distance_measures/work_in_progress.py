import math

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from log_distance_measures.config import EventLogIDs, DistanceMetric
from log_distance_measures.earth_movers_distance import earth_movers_distance


def work_in_progress_distance(
        original_log: pd.DataFrame,
        original_ids: EventLogIDs,
        simulated_log: pd.DataFrame,
        simulated_ids: EventLogIDs,
        window_size: pd.Timedelta = pd.Timedelta(hours=1),
        metric: DistanceMetric = DistanceMetric.WASSERSTEIN,
        normalize: bool = True,
) -> float:
    """
    Earth Mover's Distance (EMD) or Wasserstein Distance (1-WD) between the distribution of active cases over time.
    To get this distribution, the percentage of each window that is covered by active cases is computed. For example,
    given the window from 10am to 11am, if there are three cases active during the whole window (1 + 1 + 1), one case
    active half of the window (0.5), and two cases active a quarter of the window (0.25 + 0.25), the active value for
    that hour is 4.

    :param original_log: first event log.
    :param original_ids: mapping for the column IDs of the first event log.
    :param simulated_log: second event log.
    :param simulated_ids: mapping for the column IDs for the second event log.
    :param window_size: window to check the number of cases at the beginning of it.
    :param metric: distance metric to use in the histogram comparison (EMD or 1-WD).
    :param normalize: whether to normalize the distance metric to a value in [0.0, 1.0]

    :return: the MAE or 1-WD between the distribution of active cases over time of the two event logs, measuring the
    amount of movements (considering their distance) to transform one timestamp histogram into the other.
    """
    # Get timeline (reset to day in case daily frequency is used)
    start = min(
        original_log[original_ids.start_time].min(),
        simulated_log[simulated_ids.start_time].min()
    ).floor(freq='24h')
    end = max(
        original_log[original_ids.end_time].max(),
        simulated_log[simulated_ids.end_time].max()
    ).ceil(freq='24h')
    # Compute the active area of each bin
    original_wip = _compute_work_in_progress(original_log, original_ids, start, end, window_size)
    simulated_wip = _compute_work_in_progress(simulated_log, simulated_ids, start, end, window_size)
    # Compute EMD or 1-WD over the histograms
    if metric == DistanceMetric.EMD:
        distance = earth_movers_distance(original_wip, simulated_wip) / sum(original_wip.values())
    else:
        # Transform to 1D array
        original_1d = [element for key in original_wip for element in [key] * int(original_wip[key] * 100)]
        simulated_1d = [element for key in simulated_wip for element in [key] * int(simulated_wip[key] * 100)]
        # Measure 1-WD
        distance = wasserstein_distance(original_1d, simulated_1d)
    # Normalize if necessary
    if normalize:
        print("WARNING! The normalization of a Wasserstein Distance is sensitive to the range of the two samples, "
              "long samples may cause a higher reduction of the error.")
        max_value = max(max(original_wip.keys()), max(simulated_wip.keys()))
        distance = distance / max_value if max_value > 0 else distance
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
    Compute, for each bin of [window_size] size within the interval from [start] to [end], the percentage of "area"
    where there was an active case. For example, given the window from 10am to 11am, if there are three cases active
    during the whole window (1 + 1 + 1), one case active half of the window (0.5), and two cases active a quarter of
    the window (0.25 + 0.25), the active value for that hour is 4.

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

