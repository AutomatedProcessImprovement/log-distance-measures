import datetime

import pandas as pd

from log_distance_measures.config import DEFAULT_CSV_IDS
from log_distance_measures.remaining_time_distribution import remaining_time_distribution_distance


def _read_event_log(path: str) -> pd.DataFrame:
    event_log = pd.read_csv(path)
    event_log['start_time'] = pd.to_datetime(event_log['start_time'], utc=True)
    event_log['end_time'] = pd.to_datetime(event_log['end_time'], utc=True)
    return event_log


def test_remaining_time_distribution_similar_logs():
    # Read event log
    reference_point = pd.Timestamp("2006-11-07 12:40:00+02:00")
    event_log_1 = _read_event_log("./tests/assets/test_event_log_4.csv")
    event_log_1 = event_log_1[event_log_1[DEFAULT_CSV_IDS.end_time] > reference_point]
    event_log_1.loc[
        event_log_1[DEFAULT_CSV_IDS.start_time] < reference_point,
        DEFAULT_CSV_IDS.start_time
    ] = reference_point
    # Make a copy updating case ID, activity name, and resources, keeping timestamps
    event_log_2 = event_log_1.copy()
    event_log_2[DEFAULT_CSV_IDS.case] = event_log_2[DEFAULT_CSV_IDS.case].apply(
        lambda case_id: case_id + "_2"
    )
    event_log_2[DEFAULT_CSV_IDS.resource] = event_log_2[DEFAULT_CSV_IDS.resource].apply(
        lambda case_id: case_id + "_2"
    )
    event_log_2[DEFAULT_CSV_IDS.activity] = event_log_2[DEFAULT_CSV_IDS.activity].apply(
        lambda case_id: case_id + "_2"
    )
    # Normalized distance should be 0 as both distributions are exactly the same
    assert remaining_time_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS,
        reference_point=reference_point, bin_size=datetime.timedelta(hours=0.5), normalize=True
    ) == 0.0
    assert remaining_time_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS,
        reference_point=reference_point, bin_size=datetime.timedelta(hours=1), normalize=True
    ) == 0.0
    assert remaining_time_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS,
        reference_point=reference_point, bin_size=datetime.timedelta(hours=4), normalize=True
    ) == 0.0
    # Non normalized distance should be 0 as both distributions are exactly the same
    assert remaining_time_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS,
        reference_point=reference_point, bin_size=datetime.timedelta(hours=0.5)
    ) == 0.0
    assert remaining_time_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS,
        reference_point=reference_point, bin_size=datetime.timedelta(hours=1)
    ) == 0.0
    assert remaining_time_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS,
        reference_point=reference_point, bin_size=datetime.timedelta(hours=4)
    ) == 0.0


def test_remaining_time_distribution_shifted_logs():
    # Read event log
    reference_point = pd.Timestamp("2006-11-07 12:40:00+02:00")
    event_log_1 = _read_event_log("./tests/assets/test_event_log_4.csv")
    event_log_1 = event_log_1[event_log_1[DEFAULT_CSV_IDS.end_time] > reference_point]
    event_log_1.loc[
        event_log_1[DEFAULT_CSV_IDS.start_time] < reference_point,
        DEFAULT_CSV_IDS.start_time
    ] = reference_point
    # Make a copy updating case ID, activity name, and resources, keeping timestamps
    event_log_2 = event_log_1.copy()
    event_log_2[DEFAULT_CSV_IDS.case] = event_log_2[DEFAULT_CSV_IDS.case].apply(
        lambda case_id: case_id + "_2"
    )
    event_log_2[DEFAULT_CSV_IDS.resource] = event_log_2[DEFAULT_CSV_IDS.resource].apply(
        lambda case_id: case_id + "_2"
    )
    event_log_2[DEFAULT_CSV_IDS.activity] = event_log_2[DEFAULT_CSV_IDS.activity].apply(
        lambda case_id: case_id + "_2"
    )
    # Shift all timestamps by 3 hours
    event_log_2[DEFAULT_CSV_IDS.start_time] = event_log_2[DEFAULT_CSV_IDS.start_time] + pd.Timedelta(hours=3)
    event_log_2[DEFAULT_CSV_IDS.end_time] = event_log_2[DEFAULT_CSV_IDS.end_time] + pd.Timedelta(hours=3)
    # Normalized distance should be positive
    assert remaining_time_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS,
        reference_point=reference_point, bin_size=datetime.timedelta(hours=0.5), normalize=True
    ) > 0.0
    assert remaining_time_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS,
        reference_point=reference_point, bin_size=datetime.timedelta(hours=1), normalize=True
    ) > 0.0
    # Non normalized distance should be 3 hours (changing value depending on bin)
    assert remaining_time_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS,
        reference_point=reference_point, bin_size=datetime.timedelta(hours=0.5)
    ) == 6
    assert remaining_time_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS,
        reference_point=reference_point, bin_size=datetime.timedelta(hours=1)
    ) == 3
    assert remaining_time_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS,
        reference_point=reference_point, bin_size=datetime.timedelta(hours=3)
    ) == 1
    assert remaining_time_distribution_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS,
        reference_point=reference_point, bin_size=datetime.timedelta(hours=6)
    ) == 0.0
