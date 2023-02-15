import datetime

import pandas as pd

from log_similarity_metrics.config import DEFAULT_CSV_IDS
from log_similarity_metrics.cycle_time_distribution import cycle_time_distribution_distance


def _read_event_log(path: str) -> pd.DataFrame:
    event_log = pd.read_csv(path)
    event_log['start_time'] = pd.to_datetime(event_log['start_time'], utc=True)
    event_log['end_time'] = pd.to_datetime(event_log['end_time'], utc=True)
    return event_log


def test_cycle_time_distribution_similar_logs():
    # Read event logs with similar timestamp distribution but different resources, activity names and trace IDs
    event_log_1 = _read_event_log("./tests/assets/test_event_log_1.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_2.csv")
    # Normalized distance should be 0 as both distributions are exactly the same
    assert cycle_time_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, bin_size=datetime.timedelta(hours=0.5)
    ) == 0.0
    assert cycle_time_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, bin_size=datetime.timedelta(hours=1)
    ) == 0.0
    assert cycle_time_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, bin_size=datetime.timedelta(hours=4)
    ) == 0.0
    # Non normalized distance should be 0 as both distributions are exactly the same
    assert cycle_time_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, bin_size=datetime.timedelta(hours=0.5), normalize=False
    ) == 0.0
    assert cycle_time_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, bin_size=datetime.timedelta(hours=1), normalize=False
    ) == 0.0
    assert cycle_time_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, bin_size=datetime.timedelta(hours=4), normalize=False
    ) == 0.0


def test_cycle_time_distribution_different_logs():
    # Read event logs with different cycle times
    event_log_1 = _read_event_log("./tests/assets/test_event_log_4.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_5.csv")
    # Normalized distance should be positive as both cycle times are different (except for buckets of 4h, then it's the same)
    norm_distance = cycle_time_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, bin_size=datetime.timedelta(hours=0.5)
    )
    assert norm_distance > 0.0
    assert norm_distance < 1.0
    norm_distance = cycle_time_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, bin_size=datetime.timedelta(hours=1)
    )
    assert norm_distance > 0.0
    assert norm_distance < 1.0
    norm_distance = cycle_time_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, bin_size=datetime.timedelta(hours=4)
    )
    assert norm_distance == 0.0
    # Non normalized distance should be 0 as both cycle times are different (except for buckets of 4h, then it's the same)
    assert cycle_time_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, bin_size=datetime.timedelta(hours=0.5), normalize=False
    ) > 0.0
    assert cycle_time_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, bin_size=datetime.timedelta(hours=1), normalize=False
    ) > 0.0
    assert cycle_time_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, bin_size=datetime.timedelta(hours=4), normalize=False
    ) == 0.0
