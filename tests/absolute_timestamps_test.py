import pandas as pd

from log_similarity_metrics.absolute_timestamps import absolute_timestamps_emd, discretize_to_hour, discretize_to_day, \
    absolute_timestamps_dtw
from log_similarity_metrics.config import DEFAULT_CSV_IDS, AbsoluteTimestampType


def _read_event_log(path: str) -> pd.DataFrame:
    event_log = pd.read_csv(path)
    event_log['start_time'] = pd.to_datetime(event_log['start_time'], utc=True)
    event_log['end_time'] = pd.to_datetime(event_log['end_time'], utc=True)
    return event_log


def test_absolute_timestamps_emd_similar_logs():
    # Read event logs with similar timestamp distribution but different resources, activity names and trace IDs
    event_log_1 = _read_event_log("./tests/assets/test_event_log_1.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_2.csv")
    # EMD should be 0 as both distributions are exactly the same
    assert absolute_timestamps_emd(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH) == 0.0
    assert absolute_timestamps_emd(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START) == 0.0
    assert absolute_timestamps_emd(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END) == 0.0
    assert absolute_timestamps_emd(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH, discretize_to_hour
    ) == 0.0
    assert absolute_timestamps_emd(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START, discretize_to_hour
    ) == 0.0
    assert absolute_timestamps_emd(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END, discretize_to_hour
    ) == 0.0
    assert absolute_timestamps_emd(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH, discretize_to_day
    ) == 0.0
    assert absolute_timestamps_emd(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START, discretize_to_day
    ) == 0.0
    assert absolute_timestamps_emd(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END, discretize_to_day
    ) == 0.0


def test_absolute_timestamps_dtw_similar_logs():
    # Read event logs with similar timestamp distribution but different resources, activity names and trace IDs
    event_log_1 = _read_event_log("./tests/assets/test_event_log_1.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_2.csv")
    # DTW should be 0 as both distributions are exactly the same
    assert absolute_timestamps_dtw(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH) == 0.0
    assert absolute_timestamps_dtw(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START) == 0.0
    assert absolute_timestamps_dtw(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END) == 0.0
    assert absolute_timestamps_dtw(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH, discretize_to_hour
    ) == 0.0
    assert absolute_timestamps_dtw(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START, discretize_to_hour
    ) == 0.0
    assert absolute_timestamps_dtw(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END, discretize_to_hour
    ) == 0.0
    assert absolute_timestamps_dtw(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.BOTH, discretize_to_day
    ) == 0.0
    assert absolute_timestamps_dtw(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.START, discretize_to_day
    ) == 0.0
    assert absolute_timestamps_dtw(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, AbsoluteTimestampType.END, discretize_to_day
    ) == 0.0
