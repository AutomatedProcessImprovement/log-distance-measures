import math
import multiprocessing
import string
from concurrent.futures import ProcessPoolExecutor
from statistics import mean

import numpy as np
import pandas as pd
from jellyfish import damerau_levenshtein_distance
from scipy.optimize import linear_sum_assignment

from log_similarity_metrics.config import EventLogIDs


def control_flow_log_distance(
        original_log: pd.DataFrame,
        original_ids: EventLogIDs,
        simulated_log: pd.DataFrame,
        simulated_ids: EventLogIDs,
        parallel: bool = True
) -> float:
    """
    Compute the Control-Flow Log Distance (see "Camargo M, Dumas M, González-Rojas O. 2021. Discovering generative models
    from event logs: data-driven simulation vs deep learning. PeerJ Computer Science 7:e577 https://doi.org/10.7717/peerj-cs.577"
    for a detailed description of a similarity version of the metric).

    :param original_log: first event log.
    :param original_ids: mapping for the column IDs of the first event log.
    :param simulated_log: second event log.
    :param simulated_ids: mapping for the column IDs for the second event log.
    :param parallel: whether to run the distance computation in parallel or in one single core.

    :return: the Control-Flow Log Distance measure between [event_log_1] and [event_log_2].
    """
    # Transform the event log to a list of character sequences representing the traces
    original_sequences, simulated_sequences = _event_logs_to_activity_sequences(original_log, original_ids, simulated_log, simulated_ids)
    original_sequences.reset_index(drop=True, inplace=True)
    simulated_sequences.reset_index(drop=True, inplace=True)
    # Calculate the DL distance between each pair of traces
    cost_matrix = _compute_distance_matrix(original_sequences, simulated_sequences, parallel)
    # Get the optimum pairing
    row_indexes, col_indexes = linear_sum_assignment(cost_matrix)
    # Compute the Control-Flow Log Distance
    cfld = mean([cost_matrix[i_1, i_2] for i_1, i_2 in zip(row_indexes, col_indexes)])
    # Return the mean of the distances between the traces
    return cfld


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
    for case_id, events in event_log.groupby(log_ids.case):
        case_ids += [case_id]
        sorted_events = events.sort_values([log_ids.start_time, log_ids.end_time])
        activity_sequences += [
            "".join([mapping[activity] for activity in sorted_events[log_ids.activity]])
        ]
    # Return DataFrame with the mapped activity sequences
    return pd.DataFrame(data={'sequence': activity_sequences}, index=case_ids)


def _compute_distance_matrix(sequences_1: pd.DataFrame, sequences_2: pd.DataFrame, parallel: bool = False) -> np.ndarray:
    """
    Compute the matrix of (string edit) distances between all the sequences in [sequences_1] and [sequences_2].

    :param sequences_1: first list of sequences.
    :param sequences_2: second list of sequences.
    :return: matrix of (string edit) distances between each sequence in [sequences_1] and each sequence in [sequences_2].
    """
    # Compute distances
    num_cores = multiprocessing.cpu_count()
    num_cores = num_cores - 1 if num_cores > 2 else num_cores
    if parallel and num_cores > 1:
        # Parallel computation
        num_traces = len(sequences_1)
        traces_per_work = math.ceil(num_traces / num_cores)
        splits = [
            (i * traces_per_work, min(((i + 1) * traces_per_work), num_traces))
            for i in range(num_cores) if i * traces_per_work < num_traces
        ]
        # Run in parallel
        with ProcessPoolExecutor(max_workers=num_cores) as pool:
            args = [
                (sequences_1[start:end],
                 sequences_2,
                 start)
                for (start, end) in splits
            ]
            comparison_results = pool.map(_compare_traces_parallel, args)
        # Save results
        distances = [distance for distances in comparison_results for distance in distances]
    else:
        # Compute in one thread
        distances = _compare_traces_parallel((sequences_1, sequences_2, 0))
    # Create matrix
    distance_matrix = pd.DataFrame(distances).set_index(['i', 'j'])
    # Return numpy
    return distance_matrix.unstack().to_numpy()


def _compare_traces_parallel(args) -> list:
    # Parse args
    sequences_1, sequences_2, start_index = args
    # Compute distances
    distances = []
    for i, sequence_1 in sequences_1['sequence'].items():
        for j, sequence_2 in sequences_2['sequence'].items():
            distance = damerau_levenshtein_distance(sequence_1, sequence_2) / max(len(sequence_1), len(sequence_2))
            distances += [{'i': i + start_index, 'j': j, 'distance': distance}]
    # Return computed distances
    return distances
