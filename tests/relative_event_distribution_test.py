import pandas as pd

from log_similarity_metrics.absolute_event_distribution import discretize_to_hour, discretize_to_day, discretize_to_minute
from log_similarity_metrics.config import DEFAULT_CSV_IDS, AbsoluteTimestampType
from log_similarity_metrics.relative_event_distribution import relative_event_distribution_distance, _relativize_and_discretize


def _read_event_log(path: str) -> pd.DataFrame:
    event_log = pd.read_csv(path)
    event_log['start_time'] = pd.to_datetime(event_log['start_time'], utc=True)
    event_log['end_time'] = pd.to_datetime(event_log['end_time'], utc=True)
    return event_log


def test_relative_event_distribution_distance_similar_logs():
    # Read event logs with similar timestamp distribution but different resources, activity names and trace IDs
    event_log_1 = _read_event_log("./tests/assets/test_event_log_1.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_2.csv")
    # Normalized distance should be 0 as both timestamp distributions are exactly the same
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH, discretize_to_hour
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START, discretize_to_hour
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END, discretize_to_hour
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH, discretize_to_day
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START, discretize_to_day
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END, discretize_to_day
    ) == 0.0
    # Non normalized distance should be 0 as both timestamp distributions are exactly the same
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH, normalize=False
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START, normalize=False
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END, normalize=False
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH, discretize_to_hour, normalize=False
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START, discretize_to_hour, normalize=False
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END, discretize_to_hour, normalize=False
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH, discretize_to_day, normalize=False
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START, discretize_to_day, normalize=False
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END, discretize_to_day, normalize=False
    ) == 0.0


def test_relative_event_distribution_distance_different_but_similar_logs():
    # Read event logs with different timestamp distribution, resources, activity names and trace IDs, but similar relative times
    event_log_1 = _read_event_log("./tests/assets/test_event_log_1.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_7.csv")
    # Normalized distance should be 0 as both relative distributions are exactly the same
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH, discretize_to_hour
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START, discretize_to_hour
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END, discretize_to_hour
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH, discretize_to_day
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START, discretize_to_day
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END, discretize_to_day
    ) == 0.0
    # Non normalized distance should be 0 as both relative distributions are exactly the same
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH, normalize=False
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START, normalize=False
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END, normalize=False
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH, discretize_to_hour, normalize=False
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START, discretize_to_hour, normalize=False
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END, discretize_to_hour, normalize=False
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH, discretize_to_day, normalize=False
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START, discretize_to_day, normalize=False
    ) == 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END, discretize_to_day, normalize=False
    ) == 0.0


def test_relative_event_distribution_distance_different_logs():
    # Read event logs with different relative times
    event_log_1 = _read_event_log("./tests/assets/test_event_log_4.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_5.csv")
    # Normalized distance should be positive 0 as both distributions are different
    norm_dist = relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH
    )
    assert norm_dist > 0.0
    assert norm_dist < 1.0
    norm_dist = relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START
    )
    assert norm_dist > 0.0
    assert norm_dist < 1.0
    norm_dist = relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END
    )
    assert norm_dist > 0.0
    assert norm_dist < 1.0
    # Non normalized distance should be positive 0 as both distributions are different
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH, normalize=False
    ) > 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START, normalize=False
    ) > 0.0
    assert relative_event_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END, normalize=False
    ) > 0.0


def test__relativize_and_discretize():
    # Read event logs with similar inter-arrival times but different timestamps, resources, activity names and trace IDs
    event_log = _read_event_log("./tests/assets/test_event_log_5.csv")
    # Get inter-arrival times
    relative_timestamps = _relativize_and_discretize(event_log, DEFAULT_CSV_IDS, discretize_instant=discretize_to_minute)
    # Inter-arrival times should be similar
    assert sorted(relative_timestamps) == sorted([
        0, 1, 1, 35, 35, 43,
        0, 22, 22, 42, 42, 80,
        0, 15, 15, 38, 38, 43,
        0, 2, 2, 41, 41, 91,
        0, 4, 4, 30, 30, 34,
        0, 17, 17, 28, 28, 37,
        0, 4, 4, 19, 19, 34,
        0, 5, 5, 45, 45, 64,
        0, 58, 58, 60, 60, 105
    ])
    # The relative and discrete (to minute) times of two logs with delayed timestamps but same relative intervals
    event_log_1 = _read_event_log("./tests/assets/test_event_log_1.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_7.csv")
    assert (_relativize_and_discretize(event_log_1, DEFAULT_CSV_IDS, discretize_instant=discretize_to_minute) ==
            _relativize_and_discretize(event_log_2, DEFAULT_CSV_IDS, discretize_instant=discretize_to_minute))
