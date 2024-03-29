import pandas as pd

from log_distance_measures.circadian_workforce_distribution import circadian_workforce_distribution_distance, \
    _discretize
from log_distance_measures.config import DEFAULT_CSV_IDS


def _read_event_log(path: str) -> pd.DataFrame:
    event_log = pd.read_csv(path)
    event_log['start_time'] = pd.to_datetime(event_log['start_time'], utc=True)
    event_log['end_time'] = pd.to_datetime(event_log['end_time'], utc=True)
    return event_log


def test_circadian_workforce_distribution_distance_similar_logs():
    # Read event logs with similar timestamp distribution but different resource names, activity names and trace IDs
    event_log_1 = _read_event_log("./tests/assets/test_event_log_1.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_2.csv")
    # Normalized distance should be 0 as both distributions are exactly the same
    assert circadian_workforce_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, normalize=True
    ) == 0.0
    # Non normalized should be 0 as both distributions are exactly the same
    assert circadian_workforce_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS
    ) == 0.0


def test_circadian_workforce_distribution_distance_different_logs():
    # Read event logs with different timestamp distribution
    event_log_1 = _read_event_log("./tests/assets/test_event_log_3.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_4.csv")
    # Normalized distance should be between 0 and 1 as distributions are different
    norm_dist = circadian_workforce_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, normalize=True
    )
    assert norm_dist > 0.0
    assert norm_dist < 1.0
    # Non normalized should be greater than 0 (but lower than 23) as distributions are different
    distance = circadian_workforce_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS
    )
    assert distance > 0.0
    assert distance <= 23.0


def test_circadian_workforce_distribution_distance_non_overlapping_logs():
    # Read event logs with timestamps in different days (non overlapping)
    event_log_1 = _read_event_log("./tests/assets/test_event_log_1.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_7.csv")
    # Normalized distance should be 6/7 as one log is Mon-Wed and the other Thu-Sat (only Sunday 0 distance)
    assert circadian_workforce_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, normalize=True
    ) - 6 / 7 < 0.0001
    # Non normalized should be greater than 0 (but lower than 23) as distributions are in different days
    assert circadian_workforce_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS
    ) == 23 * 6 / 7


def test__discretize():
    event_log = _read_event_log("./tests/assets/test_event_log_9.csv")
    observations = _discretize(event_log, DEFAULT_CSV_IDS)
    workforce = observations.set_index(['weekday', 'hour'])['workforce'].to_dict()
    assert workforce == {
        # Mondays
        (0, 6): 3.0, (0, 7): 3.5, (0, 8): 3.0, (0, 9): 3.0, (0, 10): 3.0, (0, 11): 5.0,
        (0, 14): 2.0, (0, 15): 2.0, (0, 16): 3.0, (0, 17): 2.5,
        # Tuesdays
        (1, 6): 3.0, (1, 7): 4.0, (1, 8): 3.0, (1, 9): 3.0, (1, 10): 3.0, (1, 11): 5.0,
        (1, 14): 2.0, (1, 15): 2.0, (1, 16): 3.0, (1, 17): 2.0,
        # Wednesdays
        (2, 6): 3.0, (2, 7): 4.0, (2, 8): 3.0, (2, 9): 3.0, (2, 10): 3.0, (2, 11): 5.0,
        (2, 14): 2.0, (2, 15): 2.0, (2, 16): 3.0, (2, 17): 2.0,
        # Thursdays
        (3, 6): 3.0, (3, 7): 4.0, (3, 8): 3.0, (3, 9): 3.0, (3, 10): 3.0, (3, 11): 5.0,
        (3, 14): 2.0, (3, 15): 2.0, (3, 16): 3.0, (3, 17): 2.0,
        # Fridays
        (4, 6): 3.0, (4, 7): 4.0, (4, 8): 3.0, (4, 9): 3.0, (4, 10): 3.0, (4, 11): 5.0,
        (4, 14): 2.0, (4, 15): 2.0, (4, 16): 3.0, (4, 17): 2.0,
    }
