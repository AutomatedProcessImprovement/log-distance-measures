import datetime

import pandas as pd

from log_similarity_metrics.config import DEFAULT_CSV_IDS
from log_similarity_metrics.cycle_times import cycle_time_emd


def _read_event_log(path: str) -> pd.DataFrame:
    event_log = pd.read_csv(path)
    event_log['start_time'] = pd.to_datetime(event_log['start_time'], utc=True)
    event_log['end_time'] = pd.to_datetime(event_log['end_time'], utc=True)
    return event_log


def test_cycle_time_emd_similar_logs():
    # Read event logs with similar timestamp distribution but different resources, activity names and trace IDs
    event_log_1 = _read_event_log("./tests/assets/test_event_log_1.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_2.csv")
    # EMD should be 0 as both distributions are exactly the same
    assert cycle_time_emd(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, bin_size=datetime.timedelta(hours=0.5)) == 0.0
    assert cycle_time_emd(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, bin_size=datetime.timedelta(hours=1)) == 0.0
    assert cycle_time_emd(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, bin_size=datetime.timedelta(hours=4)) == 0.0
