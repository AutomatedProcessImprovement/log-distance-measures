import datetime

import pandas as pd
from pandas import Timedelta

from log_similarity_metrics.absolute_event_distribution import discretize_to_day, discretize_to_minute, discretize_to_hour
from log_similarity_metrics.case_arrival_distribution import inter_arrival_distribution_distance, _get_inter_arrival_times, \
    case_arrival_distribution_distance, _get_arrival_events
from log_similarity_metrics.config import DEFAULT_CSV_IDS


def _read_event_log(path: str) -> pd.DataFrame:
    event_log = pd.read_csv(path)
    event_log['start_time'] = pd.to_datetime(event_log['start_time'], utc=True)
    event_log['end_time'] = pd.to_datetime(event_log['end_time'], utc=True)
    return event_log


def test_case_arrival_distribution_distance_similar_logs():
    # Read event logs with same log
    event_log_1 = _read_event_log("./tests/assets/test_event_log_4.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_4.csv")
    # EMD should be 0 as both distributions are exactly the same
    assert case_arrival_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, discretize_instant=discretize_to_minute
    ) == 0.0
    assert case_arrival_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, discretize_instant=discretize_to_hour
    ) == 0.0
    assert case_arrival_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, discretize_instant=discretize_to_day
    ) == 0.0


def test_case_arrival_distribution_distance_different_logs():
    # Read event logs with same log
    event_log_1 = _read_event_log("./tests/assets/test_event_log_4.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_6.csv")
    # EMD should be 0 as both distributions are exactly the same
    assert case_arrival_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, discretize_instant=discretize_to_hour
    ) == 5 / 9
    assert case_arrival_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, discretize_instant=discretize_to_day
    ) == 0.0


def test__get_arrival_events():
    # Read event logs with similar inter-arrival times but different timestamps, resources, activity names and trace IDs
    event_log_1 = _read_event_log("./tests/assets/test_event_log_4.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_5.csv")
    # Get arrival eventsof log 1
    arrivals_1 = _get_arrival_events(event_log_1, DEFAULT_CSV_IDS)
    # The length of arrival events is similar to the length of traces in the log
    assert len(arrivals_1) == len(event_log_1[DEFAULT_CSV_IDS.case].unique())
    # There's only one arrival per trace in the log
    assert set(arrivals_1[DEFAULT_CSV_IDS.case].unique()) == set(event_log_1[DEFAULT_CSV_IDS.case].unique())
    # For each arrival event, there are no events with a previous start in the same trace
    for _, event in arrivals_1.iterrows():
        assert len(
            event_log_1[
                (event_log_1[DEFAULT_CSV_IDS.start_time] < event[DEFAULT_CSV_IDS.start_time]) &
                (event_log_1[DEFAULT_CSV_IDS.case] == event[DEFAULT_CSV_IDS.case])
                ]
        ) == 0
    # Get arrival events of log 2
    arrivals_2 = _get_arrival_events(event_log_2, DEFAULT_CSV_IDS)
    # The length of arrival events is similar to the length of traces in the log
    assert len(arrivals_2) == len(event_log_2[DEFAULT_CSV_IDS.case].unique())
    # There's only one arrival per trace in the log
    assert set(arrivals_2[DEFAULT_CSV_IDS.case].unique()) == set(event_log_2[DEFAULT_CSV_IDS.case].unique())
    # For each arrival event, there are no events with a previous start in the same trace
    for _, event in arrivals_2.iterrows():
        assert len(
            event_log_2[
                (event_log_2[DEFAULT_CSV_IDS.start_time] < event[DEFAULT_CSV_IDS.start_time]) &
                (event_log_2[DEFAULT_CSV_IDS.case] == event[DEFAULT_CSV_IDS.case])
                ]
        ) == 0
    # Both arrival events are different
    assert not arrivals_1.equals(arrivals_2)


def test_inter_arrival_distribution_distance_similar_logs():
    # Read event logs with similar inter-arrival times but different timestamps, resources, activity names and trace IDs
    event_log_1 = _read_event_log("./tests/assets/test_event_log_4.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_5.csv")
    # EMD should be 0 as both distributions are exactly the same
    assert inter_arrival_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, bin_size=datetime.timedelta(minutes=10)
    ) == 0.0
    assert inter_arrival_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, bin_size=datetime.timedelta(minutes=30)
    ) == 0.0
    assert inter_arrival_distribution_distance(
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
