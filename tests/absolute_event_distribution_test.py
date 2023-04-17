import pandas as pd

from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance
from log_distance_measures.config import DEFAULT_CSV_IDS, AbsoluteTimestampType, discretize_to_hour, discretize_to_day


def _read_event_log(path: str) -> pd.DataFrame:
    event_log = pd.read_csv(path)
    event_log['start_time'] = pd.to_datetime(event_log['start_time'], utc=True)
    event_log['end_time'] = pd.to_datetime(event_log['end_time'], utc=True)
    return event_log


def test_absolute_event_distribution_distance_similar_logs():
    # Read event logs with similar timestamp distribution but different resources, activity names and trace IDs
    event_log_1 = _read_event_log("./tests/assets/test_event_log_1.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_2.csv")
    # Normalized distance should be 0 as both distributions are exactly the same
    assert absolute_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH, normalize=True
    ) == 0.0
    assert absolute_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START, normalize=True
    ) == 0.0
    assert absolute_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END, normalize=True
    ) == 0.0
    assert absolute_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH, discretize_to_hour, normalize=True
    ) == 0.0
    assert absolute_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START, discretize_to_hour, normalize=True
    ) == 0.0
    assert absolute_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END, discretize_to_hour, normalize=True
    ) == 0.0
    assert absolute_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH, discretize_to_day, normalize=True
    ) == 0.0
    assert absolute_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START, discretize_to_day, normalize=True
    ) == 0.0
    assert absolute_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END, discretize_to_day, normalize=True
    ) == 0.0
    # Distance should be 0 as both distributions are exactly the same
    assert absolute_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH
    ) == 0.0
    assert absolute_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START
    ) == 0.0
    assert absolute_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END
    ) == 0.0
    assert absolute_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH, discretize_to_hour
    ) == 0.0
    assert absolute_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START, discretize_to_hour
    ) == 0.0
    assert absolute_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END, discretize_to_hour
    ) == 0.0
    assert absolute_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH, discretize_to_day
    ) == 0.0
    assert absolute_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START, discretize_to_day
    ) == 0.0
    assert absolute_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END, discretize_to_day
    ) == 0.0


def test_absolute_event_distribution_distance_different_logs():
    # Read event logs with similar timestamp distribution but different resources, activity names and trace IDs
    event_log_1 = _read_event_log("./tests/assets/test_event_log_1.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_7.csv")
    # Normalized distance should be between 0 and 1 as both distributions are different
    norm_dist = absolute_event_distribution_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, normalize=True)
    assert norm_dist > 0.0
    assert norm_dist < 1.0
    # Non normalized distance should be positive as both distributions are different
    norm_dist = absolute_event_distribution_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS)
    assert norm_dist > 0.0
