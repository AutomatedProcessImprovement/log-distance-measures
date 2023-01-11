import datetime
import math

import pandas as pd
from scipy.stats import wasserstein_distance

from log_similarity_metrics.config import EventLogIDs


def inter_arrival_time_emd(
        event_log_1: pd.DataFrame,
        log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        log_2_ids: EventLogIDs,
        bin_size: datetime.timedelta
) -> float:
    """
    EMD (or Wasserstein Distance) between the distribution of inter-arrival times of two event logs. To get this distribution, the
    inter-arrival times are discretized to bins of size [bin_size].

    :param event_log_1: first event log.
    :param log_1_ids: mapping for the column IDs of the first event log.
    :param event_log_2: second event log.
    :param log_2_ids: mapping for the column IDs for the second event log.
    :param bin_size: time interval to define the bin size.

    :return: the EMD between the inter-arrival time distribution of the two event logs, measuring the amount of movements (considering their
    distance) to transform one inter-arrival time histogram into the other.
    """
    # Get inter-arrival times
    inter_arrivals_1 = _get_inter_arrival_times(event_log_1, log_1_ids)
    inter_arrivals_2 = _get_inter_arrival_times(event_log_2, log_2_ids)
    # Discretize each instant to its corresponding "bin"
    discretized_inter_arrivals_1 = [math.floor(inter_arrival / bin_size) for inter_arrival in inter_arrivals_1]
    discretized_inter_arrivals_2 = [math.floor(inter_arrival / bin_size) for inter_arrival in inter_arrivals_2]
    # Return EMD metric
    return wasserstein_distance(discretized_inter_arrivals_1, discretized_inter_arrivals_2)


def _get_inter_arrival_times(event_log: pd.DataFrame, log_ids: EventLogIDs) -> list:
    # Get absolute arrivals
    arrivals = []
    for case, events in event_log.groupby([log_ids.case]):
        arrivals += [events[log_ids.start_time].min()]
    arrivals.sort()
    # Compute times between each arrival and the next one
    inter_arrivals = []
    last_arrival = None
    for arrival in arrivals:
        if last_arrival:
            inter_arrivals += [arrival - last_arrival]
        last_arrival = arrival
    # Return list with inter-arrivals
    return inter_arrivals
