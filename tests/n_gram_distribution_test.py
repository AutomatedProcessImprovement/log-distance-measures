import pandas as pd

from log_similarity_metrics.config import DEFAULT_CSV_IDS
from log_similarity_metrics.n_gram_distribution import _compute_n_grams, n_gram_distribution_distance


def _read_event_log(path: str) -> pd.DataFrame:
    event_log = pd.read_csv(path)
    event_log['start_time'] = pd.to_datetime(event_log['start_time'], utc=True)
    event_log['end_time'] = pd.to_datetime(event_log['end_time'], utc=True)
    return event_log


def test__compute_n_grams():
    # Read event log
    # Mapping -> [None, 'A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'G']
    # Traces: ABCDEFHI / ABDCEFHI / ABCDEGHI / ABDCEGHI
    event_log = _read_event_log("./tests/assets/test_event_log_1.csv")
    activity_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'G']
    # Compute 2-gram
    bigrams = _compute_n_grams(event_log, DEFAULT_CSV_IDS, activity_labels, 2)
    assert bigrams == {
        '0,1': 4, '1,2': 4, '2,3': 2, '3,4': 2, '4,5': 2, '5,6': 2, '6,7': 2,
        '7,8': 4, '8,0': 4, '2,4': 2, '4,3': 2, '3,5': 2, '5,9': 2, '9,7': 2,
    }
    # Compute 3-gram
    trigrams = _compute_n_grams(event_log, DEFAULT_CSV_IDS, activity_labels, 3)
    assert trigrams == {
        '0,0,1': 4, '0,1,2': 4, '1,2,3': 2, '2,3,4': 2, '3,4,5': 2, '4,5,6': 1,
        '5,6,7': 2, '6,7,8': 2, '7,8,0': 4, '8,0,0': 4, '1,2,4': 2, '2,4,3': 2,
        '4,3,5': 2, '3,5,6': 1, '4,5,9': 1, '5,9,7': 2, '9,7,8': 2, '3,5,9': 1,
    }
    # Compute 4-gram
    cuatrigrams = _compute_n_grams(event_log, DEFAULT_CSV_IDS, activity_labels, 4)
    assert cuatrigrams == {
        '0,0,0,1': 4, '0,0,1,2': 4, '8,0,0,0': 4, '7,8,0,0': 4,
        '0,1,2,3': 2, '0,1,2,4': 2, '6,7,8,0': 2, '9,7,8,0': 2,
        '1,2,3,4': 2, '1,2,4,3': 2, '2,3,4,5': 2, '2,4,3,5': 2,
        '3,4,5,6': 1, '3,4,5,9': 1, '3,5,6,7': 1, '3,5,9,7': 1,
        '4,3,5,6': 1, '4,3,5,9': 1, '4,5,6,7': 1, '4,5,9,7': 1,
        '5,6,7,8': 2, '5,9,7,8': 2
    }


def test_n_gram_distribution_distance_similar_logs():
    # Read same event log twice
    event_log_1 = _read_event_log("./tests/assets/test_event_log_1.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_1.csv")
    # Normalized N-gram distance should be 0 as both distributions are exactly the same
    assert n_gram_distribution_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, 2) == 0.0
    assert n_gram_distribution_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, 3) == 0.0
    assert n_gram_distribution_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, 4) == 0.0
    # N-gram distance should be 0 as both distributions are exactly the same
    assert n_gram_distribution_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, 2, False) == 0.0
    assert n_gram_distribution_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, 3, False) == 0.0
    assert n_gram_distribution_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, 4, False) == 0.0


def test_n_gram_distribution_distance_disjoint_logs():
    # Read same event log twice
    event_log_1 = _read_event_log("./tests/assets/test_event_log_1.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_2.csv")
    # Normalized N-gram distance should be 1.0 as both distributions are exactly the same
    assert n_gram_distribution_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, 2) == 1.0
    assert n_gram_distribution_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, 3) == 1.0
    assert n_gram_distribution_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, 4) == 1.0
    # N-gram distance should be the highest value possible for this logs
    num_bigrams = sum(
        [len(events) + 1 for case_id, events in event_log_1.groupby(DEFAULT_CSV_IDS.case)] +
        [len(events) + 1 for case_id, events in event_log_2.groupby(DEFAULT_CSV_IDS.case)]
    )
    assert n_gram_distribution_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, 2, False) == num_bigrams
    num_trigrams = sum(
        [len(events) + 2 for case_id, events in event_log_1.groupby(DEFAULT_CSV_IDS.case)] +
        [len(events) + 2 for case_id, events in event_log_2.groupby(DEFAULT_CSV_IDS.case)]
    )
    assert n_gram_distribution_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, 3, False) == num_trigrams
    num_cuatrigrams = sum(
        [len(events) + 3 for case_id, events in event_log_1.groupby(DEFAULT_CSV_IDS.case)] +
        [len(events) + 3 for case_id, events in event_log_2.groupby(DEFAULT_CSV_IDS.case)]
    )
    assert n_gram_distribution_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, 4, False) == num_cuatrigrams


def test_n_gram_distribution_distance_different_logs():
    # Read event logs with similar timestamp distribution but different resources, activity names and trace IDs
    event_log_1 = _read_event_log("./tests/assets/test_event_log_1.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_3.csv")
    # Normalized N-gram distance should maintain the proportions
    num_bigrams = sum(
        [len(events) + 1 for case_id, events in event_log_1.groupby(DEFAULT_CSV_IDS.case)] +
        [len(events) + 1 for case_id, events in event_log_2.groupby(DEFAULT_CSV_IDS.case)]
    )
    assert n_gram_distribution_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, 2) == 12 / num_bigrams
    num_trigrams = sum(
        [len(events) + 2 for case_id, events in event_log_1.groupby(DEFAULT_CSV_IDS.case)] +
        [len(events) + 2 for case_id, events in event_log_2.groupby(DEFAULT_CSV_IDS.case)]
    )
    assert n_gram_distribution_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, 3) == 16 / num_trigrams
    num_cuatrigrams = sum(
        [len(events) + 3 for case_id, events in event_log_1.groupby(DEFAULT_CSV_IDS.case)] +
        [len(events) + 3 for case_id, events in event_log_2.groupby(DEFAULT_CSV_IDS.case)]
    )
    assert n_gram_distribution_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, 4) == 20 / num_cuatrigrams
    # N-gram distance should be the number of different n-grams
    assert n_gram_distribution_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, 2, False) == 12
    assert n_gram_distribution_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, 3, False) == 16
    assert n_gram_distribution_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, 4, False) == 20
