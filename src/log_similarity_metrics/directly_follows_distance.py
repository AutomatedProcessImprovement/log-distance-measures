import pandas as pd
from scipy.stats import wasserstein_distance

from log_similarity_metrics.config import EventLogIDs


def directly_follows_distance(
        event_log_1: pd.DataFrame,
        log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        log_2_ids: EventLogIDs,
        n: int = 3
) -> float:
    """
    Compute the distance between the directly-follows relations of two event logs.

    :param event_log_1: first event log.
    :param log_1_ids: mapping for the column IDs of the first event log.
    :param event_log_2: second event log.
    :param log_2_ids: mapping for the column IDs for the second event log.
    :param n: size of the n-grams to build (e.g. n=3 will compare sequences of 3 activity instances like ABC and ABD).

    :return: the Control-Flow Log Similarity measure between [event_log_1] and [event_log_2].
    """
    # Build n-grams histogram for each event log
    n_histogram_1 = _compute_n_grams(event_log_1, log_1_ids, n)
    n_histogram_2 = _compute_n_grams(event_log_2, log_2_ids, n)
    # Fill each histogram with a 0 for the n_grams missing from the other histogram
    frequencies_1, frequencies_2 = [], []
    for key in set(list(n_histogram_1.keys()) + list(n_histogram_2.keys())):
        frequencies_1 += [
            n_histogram_1[key] if key in n_histogram_1 else 0
        ]
        frequencies_2 += [
            n_histogram_2[key] if key in n_histogram_2 else 0
        ]
    # Return EMD metric TODO not penalize for distance
    return wasserstein_distance(frequencies_1, frequencies_2)


def _compute_n_grams(event_log: pd.DataFrame, log_ids: EventLogIDs, n: int = 3) -> dict:
    """
    Compute the n-grams of activities (directly-follows) of an event log.

    :param event_log: event log to analyze.
    :param log_ids: mapping for the column IDs of the event log.
    :param n: size of the n-grams to compute.

    :return: a dict with the n-grams as key, and their absolute frequency as value.
    """
    # Compute n-grams
    n_grams = {}
    # Return n_grams and their frequency
    return n_grams
