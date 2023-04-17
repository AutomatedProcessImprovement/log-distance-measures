import pandas as pd

from log_distance_measures.config import EventLogIDs


def n_gram_distribution_distance(
        original_log: pd.DataFrame,
        original_ids: EventLogIDs,
        simulated_log: pd.DataFrame,
        simulated_ids: EventLogIDs,
        n: int = 3,
        normalize: bool = True
) -> float:
    """
    Compute the distance between the frequency of n-grams in two event logs.

    :param original_log: first event log.
    :param original_ids: mapping for the column IDs of the first event log.
    :param simulated_log: second event log.
    :param simulated_ids: mapping for the column IDs for the second event log.
    :param n: size of the n-grams to build (e.g. n=3 will compare sequences of 3 activity instances like ABC and ABD).
    :param normalize: whether to normalize the distance metric to a value in [0.0, 1.0]

    :return: the sum of absolute errors between the frequency distribution of n-grams in [event_log_1] and [event_log_2].
    """
    # Map of each activity to a number
    activity_labels = list(set(
        original_log[original_ids.activity].unique().tolist() +
        simulated_log[simulated_ids.activity].unique().tolist()
    ))
    # Build n-grams histogram for each event log
    original_n_histogram = _compute_n_grams(original_log, original_ids, activity_labels, n)
    simulated_n_histogram = _compute_n_grams(simulated_log, simulated_ids, activity_labels, n)
    # Fill each histogram with a 0 for the n_grams missing from the other histogram
    original_frequencies, simulated_frequencies = [], []
    for key in set(list(original_n_histogram.keys()) + list(simulated_n_histogram.keys())):
        original_frequencies += [
            original_n_histogram[key] if key in original_n_histogram else 0
        ]
        simulated_frequencies += [
            simulated_n_histogram[key] if key in simulated_n_histogram else 0
        ]
    # Compute distance metric
    distance = sum([abs(x - y) for (x, y) in zip(original_frequencies, simulated_frequencies)])
    if normalize:
        distance = distance / (sum(original_frequencies) + sum(simulated_frequencies))
    # Return metric
    return distance


def _compute_n_grams(event_log: pd.DataFrame, log_ids: EventLogIDs, activity_labels: list, n: int = 3) -> dict:
    """
    Compute the n-grams of activities (directly-follows) of an event log.

    :param event_log: event log to analyze.
    :param log_ids: mapping for the column IDs of the event log.
    :param activity_labels: list with the unique activity labels to map them to n-grams.
    :param n: size of the n-grams to compute.

    :return: a dict with the n-grams as key, and their absolute frequency as value.
    """
    # Extend activity IDs with "None" (for start end of trace)
    activity_to_int = [None] + activity_labels
    # Compute n-grams
    n_grams = {}
    for case_id, events in event_log.groupby(log_ids.case):
        # List with the IDs of each activity
        events = [0] * (n - 1) + [
            activity_to_int.index(event) for event in events.sort_values([log_ids.start_time, log_ids.end_time])[log_ids.activity]
        ] + [0] * (n - 1)
        # Go over the IDs in a n-sized window
        for i in range(len(events) - n + 1):
            n_gram = ",".join([str(event) for event in events[i: i + n]])
            n_grams[n_gram] = n_grams[n_gram] + 1 if n_gram in n_grams else 1
    # Return n_grams and their frequency
    return n_grams
