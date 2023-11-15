import pandas as pd

from log_distance_measures.config import DEFAULT_CSV_IDS, DistanceMetric
from log_distance_measures.work_in_progress import work_in_progress_distance, _compute_work_in_progress


def _read_event_log(path: str) -> pd.DataFrame:
    event_log = pd.read_csv(path)
    event_log['start_time'] = pd.to_datetime(event_log['start_time'], utc=True)
    event_log['end_time'] = pd.to_datetime(event_log['end_time'], utc=True)
    return event_log


def test_work_in_progress_distance_similar_logs():
    # Read event logs with same log
    event_log_1 = _read_event_log("./tests/assets/test_event_log_4.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_4.csv")
    # Normalized and non-normalized distance should be 0 as both distributions are exactly the same
    assert work_in_progress_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS) == 0.0
    assert work_in_progress_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, window_size=pd.Timedelta(minutes=1)
    ) == 0.0
    assert work_in_progress_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, normalize=True) == 0.0
    assert work_in_progress_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, window_size=pd.Timedelta(minutes=1), normalize=True
    ) == 0.0


def test_work_in_progress_distance_different_logs():
    # Read event logs with different logs
    event_log_1 = _read_event_log("./tests/assets/test_event_log_4.csv")
    event_log_2 = _read_event_log("./tests/assets/test_event_log_6.csv")
    # Normalized distance should be positive as both distributions have differences
    norm_dist = work_in_progress_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, normalize=True
    )
    assert norm_dist > 0.0
    assert norm_dist < 1.0
    # Non normalized distance should be positive as both distributions have differences
    assert work_in_progress_distance(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS) > 0.0
    # Wasserstein should be 0 for days because there's only one bin, so both have same density function
    assert work_in_progress_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS, window_size=pd.Timedelta(days=1)
    ) == 0
    # EMD should be >0 for days, because they have different mass (small penalization for extra elements)
    assert work_in_progress_distance(
        event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS,
        window_size=pd.Timedelta(days=1), metric=DistanceMetric.EMD
    ) > 0.0


def test__compute_work_in_progress():
    event_log = _read_event_log("./tests/assets/test_event_log_4.csv")
    start = event_log[DEFAULT_CSV_IDS.start_time].min().floor(freq="H")
    end = event_log[DEFAULT_CSV_IDS.end_time].max().ceil(freq="H")
    assert _compute_work_in_progress(event_log, DEFAULT_CSV_IDS, start, end, pd.Timedelta(hours=1)) == {
        0: 0.8333333333333334,
        1: 3.816666666666667,
        2: 5.083333333333333,
        3: 2.716666666666667,
        4: 0.7166666666666667
    }
    event_log = _read_event_log("./tests/assets/test_event_log_8.csv")
    start = event_log[DEFAULT_CSV_IDS.start_time].min().floor(freq="H")
    end = event_log[DEFAULT_CSV_IDS.end_time].max().ceil(freq="H")
    assert _compute_work_in_progress(event_log, DEFAULT_CSV_IDS, start, end, pd.Timedelta(hours=1)) == {
        0: 0.5,
        1: 2.5,
        2: 2.1333333333333333,
        3: 3.1333333333333333,
        4: 3.1166666666666667,
        5: 2,
        6: 0.016666666666666666
    }
