import pandas as pd

from log_distance_measures.circadian_event_distribution import circadian_event_distribution_distance
from log_distance_measures.config import DEFAULT_CSV_IDS, AbsoluteTimestampType


def _read_event_log(path: str) -> pd.DataFrame:
    event_log = pd.read_csv(path)
    event_log['start_time'] = pd.to_datetime(event_log['start_time'], utc=True)
    event_log['end_time'] = pd.to_datetime(event_log['end_time'], utc=True)
    return event_log


def test_circadian_event_distribution_distance_similar_logs():
    # Read event logs with similar timestamp distribution but different resources, activity names and trace IDs
    event_log_1 = _read_event_log("./tests/assets/test_event_log_1.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_2.csv")
    # Normalized distance should be 0 as both distributions are exactly the same
    assert circadian_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH, normalize=True
    ) == 0.0
    assert circadian_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START, normalize=True
    ) == 0.0
    assert circadian_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END, normalize=True
    ) == 0.0
    # Non normalized should be 0 as both distributions are exactly the same
    assert circadian_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH
    ) == 0.0
    assert circadian_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START
    ) == 0.0
    assert circadian_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END
    ) == 0.0


def test_circadian_event_distribution_distance_different_logs():
    # Read event logs with different timestamp distribution
    event_log_1 = _read_event_log("./tests/assets/test_event_log_3.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_4.csv")
    # Normalized distance should be between 0 and 1 as distributions are different
    norm_dist = circadian_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH, normalize=True
    )
    assert norm_dist > 0.0
    assert norm_dist < 1.0
    norm_dist = circadian_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START, normalize=True
    )
    assert norm_dist > 0.0
    assert norm_dist < 1.0
    norm_dist = circadian_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END, normalize=True
    )
    assert norm_dist > 0.0
    assert norm_dist < 1.0
    # Non normalized should be greater than 0 (but lower than 23) as distributions are different
    assert circadian_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH
    ) > 0.0
    assert circadian_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START
    ) > 0.0
    assert circadian_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END
    ) > 0.0


def test_circadian_event_distribution_distance_non_overlapping_logs():
    # Read event logs with timestamps in different days (non overlapping)
    event_log_1 = _read_event_log("./tests/assets/test_event_log_1.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_7.csv")
    # Normalized distance should be 6/7 as one log is Mon-Wed and the other Thu-Sat (only Sunday 0 distance)
    assert circadian_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH, normalize=True
    ) - 6 / 7 < 0.0001
    assert circadian_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START, normalize=True
    ) - 6 / 7 < 0.0001
    assert circadian_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END, normalize=True
    ) - 6 / 7 < 0.0001
    # Non normalized should be greater than 0 (but lower than 23) as distributions are in different days
    assert circadian_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH
    ) == 23 * 6 / 7
    assert circadian_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START
    ) == 23 * 6 / 7
    assert circadian_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END
    ) == 23 * 6 / 7
