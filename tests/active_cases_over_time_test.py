import pandas as pd

from log_similarity_metrics.active_cases_over_time import active_cases_over_time_distance, _active_cases_over_time
from log_similarity_metrics.config import DEFAULT_CSV_IDS


def _read_event_log(path: str) -> pd.DataFrame:
    event_log = pd.read_csv(path)
    event_log['start_time'] = pd.to_datetime(event_log['start_time'], utc=True)
    event_log['end_time'] = pd.to_datetime(event_log['end_time'], utc=True)
    return event_log


def test_active_cases_over_time_distance_similar_logs():
    # Read event logs with same log
    event_log_1 = _read_event_log("./tests/assets/test_event_log_4.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_4.csv")
    # Normalized distance should be 0 as both distributions are exactly the same
    assert active_cases_over_time_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS) == 0.0
    assert active_cases_over_time_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, window_size=pd.Timedelta(minutes=1)
    ) == 0.0
    assert active_cases_over_time_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, normalize=True) == 0.0
    assert active_cases_over_time_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, window_size=pd.Timedelta(minutes=1), normalize=True
    ) == 0.0


def test_active_cases_over_time_distance_different_logs():
    # Read event logs with same log
    event_log_1 = _read_event_log("./tests/assets/test_event_log_4.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_6.csv")
    # Normalized distance should be positive by hour but 0 by day as both distributions have differences in the hour scale
    norm_dist = active_cases_over_time_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, normalize=True
    )
    assert norm_dist > 0.0
    assert norm_dist < 1.0
    assert active_cases_over_time_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, window_size=pd.Timedelta(days=1), normalize=True
    ) == 0.0
    # Non normalized distance should be positive by hour but 0 by day as both distributions have differences in the hour scale
    assert active_cases_over_time_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS) > 0.0
    assert active_cases_over_time_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, window_size=pd.Timedelta(days=1)
    ) == 0.0


def test__active_cases_over_time():
    event_log = _read_event_log("./tests/assets/test_event_log_4.csv")
    start = event_log[DEFAULT_CSV_IDS.start_time].min().floor(freq="H")
    end = event_log[DEFAULT_CSV_IDS.end_time].max().ceil(freq="H")
    assert _active_cases_over_time(event_log, DEFAULT_CSV_IDS, start, end) == [0, 3, 5, 4, 2, 0]
    event_log = _read_event_log("./tests/assets/test_event_log_8.csv")
    start = event_log[DEFAULT_CSV_IDS.start_time].min().floor(freq="H")
    end = event_log[DEFAULT_CSV_IDS.end_time].max().ceil(freq="H")
    assert _active_cases_over_time(event_log, DEFAULT_CSV_IDS, start, end) == [0, 2, 3, 2, 4, 2, 2, 0]
