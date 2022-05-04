import string
from statistics import mean

import numpy as np
import pandas as pd
from jellyfish import damerau_levenshtein_distance
from scipy.optimize import linear_sum_assignment

from log_similarity_metrics.config import EventLogIDs


def control_flow_log_similarity(
        event_log_1: pd.DataFrame,
        log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        log_2_ids: EventLogIDs
) -> float:
    """
    Compute the Control-Flow Log Similarity (see "Camargo M, Dumas M, González-Rojas O. 2021. Discovering generative models
    from event logs: data-driven simulation vs deep learning. PeerJ Computer Science 7:e577 https://doi.org/10.7717/peerj-cs.577"
    for a detailed description of the metric).

    :param event_log_1: first event log.
    :param log_1_ids: mapping for the column IDs of the first event log.
    :param event_log_2: second event log.
    :param log_2_ids: mapping for the column IDs for the second event log.

    :return: the Control-Flow Log Similarity measure between [event_log_1] and [event_log_2].
    """
    # Transform the event log to a list of character sequences representing the traces
    sequences_1, sequences_2 = _event_logs_to_activity_sequences(event_log_1, log_1_ids, event_log_2, log_2_ids)
    # Calculate the DL distance between each pair of traces
    distances = []
    for i_1, sequence_1 in sequences_1['sequence'].iteritems():
        for i_2, sequence_2 in sequences_2['sequence'].iteritems():
            distance = damerau_levenshtein_distance(sequence_1, sequence_2) / max(len(sequence_1), len(sequence_2))
            distances += [{'i_1': i_1, 'i_2': i_2, 'distance': distance}]
    distance_matrix = pd.DataFrame(distances).set_index(['i_1', 'i_2'])
    # Get the optimum pairing
    cost_matrix = distance_matrix.unstack().to_numpy()
    row_indexes, col_indexes = linear_sum_assignment(np.array(cost_matrix))
    # Compute the Control-Flow Log Similarity
    longest_case_size = max(
        [len(events) for case_id, events in event_log_1.groupby([log_1_ids.case])] +
        [len(events) for case_id, events in event_log_2.groupby([log_2_ids.case])]
    )
    cfls = mean(
        [
            (1 - cost_matrix[i_1, i_2])  # Compute Control-Flow Trace Similarity (1 - normalized DL distance)
            for i_1, i_2 in zip(row_indexes, col_indexes)
        ]
    )
    # Return the mean of the distances between the traces
    return cfls


def _event_logs_to_activity_sequences(
        event_log_1: pd.DataFrame,
        log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        log_2_ids: EventLogIDs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transform two event logs into [pd.DataFrame]s with the case IDs as index, and the sequence of activities (mapped to single characters)
    as value.

    :param event_log_1: first event log.
    :param log_1_ids: mapping for the column IDs of the first event log.
    :param event_log_2: second event log.
    :param log_2_ids: mapping for the column IDs for the second event log.

    :return: a tuple of two [pd.DataFrame]s with the case IDs as index and its corresponding activity sequence (mapped to single characters)
             as values.
    """
    # Check if there are enough characters to map all activities
    characters = string.printable
    activities = set(list(event_log_1[log_1_ids.activity].unique()) + list(event_log_2[log_2_ids.activity].unique()))
    if len(activities) > len(characters):
        raise RuntimeError(
            "Not enough characters ({}) to map all activities ({}) in Damerau Levenshtein distance calculation!".format(
                len(characters),
                len(activities)
            )
        )
    # Declare mapping
    mapping = {activity: characters[index] for index, activity in enumerate(activities)}
    # Calculate event logs
    processed_event_log_1 = _event_log_to_activity_sequence(event_log_1, log_1_ids, mapping)
    processed_event_log_2 = _event_log_to_activity_sequence(event_log_2, log_2_ids, mapping)
    # Return value
    return processed_event_log_1, processed_event_log_2


def _event_log_to_activity_sequence(event_log: pd.DataFrame, log_ids: EventLogIDs, mapping: dict) -> pd.DataFrame:
    """
    Transform an event log into a [pd.DataFrame] with the case IDs as index, and the sequence of activities (mapped to single characters)
    as value.

    :param event_log: event log to transform.
    :param log_ids: mapping for the column IDs of the event log.
    :param mapping: mapping between activities and single characters.

    :return: a [pd.DataFrame] with the case IDs as index and its corresponding activity sequence (mapped to single characters) as values.
    """
    # Define lists to store each case
    case_ids = []
    activity_sequences = []
    # For each case, map the activities to character sequence
    for case_id, events in event_log.groupby([log_ids.case]):
        case_ids += [case_id]
        sorted_events = events.sort_values([log_ids.end_time, log_ids.start_time])
        activity_sequences += [
            "".join([mapping[activity] for activity in sorted_events[log_ids.activity]])
        ]
    # Return DataFrame with the mapped activity sequences
    return pd.DataFrame(data={'sequence': activity_sequences}, index=case_ids)
