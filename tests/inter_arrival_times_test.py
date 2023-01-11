import datetime

import pandas as pd
from pandas import Timedelta

from log_similarity_metrics.config import DEFAULT_CSV_IDS
from log_similarity_metrics.inter_arrival_times import inter_arrival_time_emd, _get_inter_arrival_times


def _read_event_log(path: str) -> pd.DataFrame:
    event_log = pd.read_csv(path)
    event_log['start_time'] = pd.to_datetime(event_log['start_time'], utc=True)
    event_log['end_time'] = pd.to_datetime(event_log['end_time'], utc=True)
    return event_log


def test_inter_arrival_time_emd_similar_logs():
    # Read event logs with similar inter-arrival times but different timestamps, resources, activity names and trace IDs
    event_log_1 = _read_event_log("./tests/assets/test_event_log_4.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_5.csv")
    # EMD should be 0 as both distributions are exactly the same
    assert inter_arrival_time_emd(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, bin_size=datetime.timedelta(minutes=10)
    ) == 0.0
    assert inter_arrival_time_emd(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, bin_size=datetime.timedelta(minutes=30)
    ) == 0.0
    assert inter_arrival_time_emd(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, bin_size=datetime.timedelta(hours=1)
    ) == 0.0


def test__get_inter_arrival_times():
    # Read event logs with similar inter-arrival times but different timestamps, resources, activity names and trace IDs
    event_log_1 = _read_event_log("./tests/assets/test_event_log_4.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_5.csv")
    # Get inter-arrival times
    inter_arrivals_1 = _get_inter_arrival_times(event_log_1, DEFAULT_CSV_IDS)
    inter_arrivals_2 = _get_inter_arrival_times(event_log_2, DEFAULT_CSV_IDS)
    # Inter-arrival times should be similar
    assert inter_arrivals_1 == inter_arrivals_2
    # Inter-arrival times should be the expected
    assert sorted(inter_arrivals_1) == [
        Timedelta(minutes=10),
        Timedelta(minutes=11),
        Timedelta(minutes=11),
        Timedelta(minutes=20),
        Timedelta(minutes=23),
        Timedelta(minutes=25),
        Timedelta(minutes=30),
        Timedelta(minutes=35)
    ]
