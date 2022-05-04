import pandas as pd

from log_similarity_metrics.config import DEFAULT_CSV_IDS
from log_similarity_metrics.control_flow_log_similarity import _event_log_to_activity_sequence, _event_logs_to_activity_sequences, \
    control_flow_log_similarity


def _read_event_log(path: str) -> pd.DataFrame:
    event_log = pd.read_csv(path)
    event_log['start_time'] = pd.to_datetime(event_log['start_time'], utc=True)
    event_log['end_time'] = pd.to_datetime(event_log['end_time'], utc=True)
    return event_log


def test_control_flow_log_similarity():
    # Define event logs
    event_log_1 = pd.DataFrame(
        data=[
            {'case_id': "trace-01", 'Activity': "A", 'start_time': 1, 'end_time': 2},
            {'case_id': "trace-01", 'Activity': "B", 'start_time': 3, 'end_time': 4},
            {'case_id': "trace-01", 'Activity': "C", 'start_time': 5, 'end_time': 6},
            {'case_id': "trace-01", 'Activity': "D", 'start_time': 7, 'end_time': 8},
            {'case_id': "trace-02", 'Activity': "A", 'start_time': 12, 'end_time': 13},
            {'case_id': "trace-02", 'Activity': "B", 'start_time': 14, 'end_time': 15},
            {'case_id': "trace-02", 'Activity': "F", 'start_time': 16, 'end_time': 17},
            {'case_id': "trace-02", 'Activity': "G", 'start_time': 18, 'end_time': 19},
            {'case_id': "trace-03", 'Activity': "A", 'start_time': 23, 'end_time': 24},
            {'case_id': "trace-03", 'Activity': "H", 'start_time': 25, 'end_time': 26},
            {'case_id': "trace-03", 'Activity': "I", 'start_time': 27, 'end_time': 28}
        ]
    )
    event_log_2 = pd.DataFrame(
        data=[
            {'case_id': "trace-01", 'Activity': "A", 'start_time': 1, 'end_time': 2},
            {'case_id': "trace-01", 'Activity': "C", 'start_time': 3, 'end_time': 4},
            {'case_id': "trace-01", 'Activity': "B", 'start_time': 5, 'end_time': 6},
            {'case_id': "trace-01", 'Activity': "D", 'start_time': 7, 'end_time': 8},
            {'case_id': "trace-02", 'Activity': "A", 'start_time': 12, 'end_time': 13},
            {'case_id': "trace-02", 'Activity': "B", 'start_time': 14, 'end_time': 15},
            {'case_id': "trace-02", 'Activity': "F", 'start_time': 16, 'end_time': 17},
            {'case_id': "trace-02", 'Activity': "G", 'start_time': 18, 'end_time': 19},
            {'case_id': "trace-03", 'Activity': "A", 'start_time': 23, 'end_time': 24},
            {'case_id': "trace-03", 'Activity': "J", 'start_time': 25, 'end_time': 26},
            {'case_id': "trace-03", 'Activity': "I", 'start_time': 27, 'end_time': 28},
            {'case_id': "trace-03", 'Activity': "H", 'start_time': 29, 'end_time': 30},
            {'case_id': "trace-03", 'Activity': "H", 'start_time': 32, 'end_time': 35}
        ]
    )
    # Get log similarity
    cfls = control_flow_log_similarity(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS)
    # Check result
    assert cfls == (2.2 / 3)


def test__event_logs_to_activity_sequences():
    # Define event logs
    event_log_1 = pd.DataFrame(
        data=[
            {'case_id': "trace-01", 'Activity': "Start", 'start_time': 1, 'end_time': 2},
            {'case_id': "trace-01", 'Activity': "End", 'start_time': 3, 'end_time': 4},
            {'case_id': "trace-02", 'Activity': "Start", 'start_time': 5, 'end_time': 6},
            {'case_id': "trace-02", 'Activity': "Do something", 'start_time': 7, 'end_time': 9},
            {'case_id': "trace-02", 'Activity': "Do another something", 'start_time': 10, 'end_time': 11},
            {'case_id': "trace-02", 'Activity': "End", 'start_time': 11, 'end_time': 13},
            {'case_id': "trace-03", 'Activity': "Start", 'start_time': 23, 'end_time': 24}
        ]
    )
    event_log_2 = pd.DataFrame(
        data=[
            {'case_id': "trace-01", 'Activity': "Start", 'start_time': 1, 'end_time': 2},
            {'case_id': "trace-02", 'Activity': "Start", 'start_time': 23, 'end_time': 24},
            {'case_id': "trace-02", 'Activity': "Do another something", 'start_time': 30, 'end_time': 40},
            {'case_id': "trace-03", 'Activity': "Start", 'start_time': 5, 'end_time': 6},
            {'case_id': "trace-03", 'Activity': "Do something", 'start_time': 7, 'end_time': 8},
            {'case_id': "trace-03", 'Activity': "Do another something", 'start_time': 9, 'end_time': 10},
            {'case_id': "trace-03", 'Activity': "End", 'start_time': 11, 'end_time': 13}
        ]
    )
    # Transform to sequences
    sequence_log_1, sequence_log_2 = _event_logs_to_activity_sequences(event_log_1, DEFAULT_CSV_IDS, event_log_2, DEFAULT_CSV_IDS)
    # Check if the length of the traces is the same of the length of the string representing the sequence of activities
    assert len(sequence_log_1.loc['trace-01']['sequence']) == len(event_log_1[event_log_1[DEFAULT_CSV_IDS.case] == 'trace-01'])
    assert len(sequence_log_1.loc['trace-02']['sequence']) == len(event_log_1[event_log_1[DEFAULT_CSV_IDS.case] == 'trace-02'])
    assert len(sequence_log_1.loc['trace-03']['sequence']) == len(event_log_1[event_log_1[DEFAULT_CSV_IDS.case] == 'trace-03'])
    assert len(sequence_log_2.loc['trace-01']['sequence']) == len(event_log_2[event_log_2[DEFAULT_CSV_IDS.case] == 'trace-01'])
    assert len(sequence_log_2.loc['trace-02']['sequence']) == len(event_log_2[event_log_2[DEFAULT_CSV_IDS.case] == 'trace-02'])
    assert len(sequence_log_2.loc['trace-03']['sequence']) == len(event_log_2[event_log_2[DEFAULT_CSV_IDS.case] == 'trace-03'])
    # Check if same traces have the same sequences
    assert sequence_log_1.loc['trace-03']['sequence'] == sequence_log_2.loc['trace-01']['sequence']
    assert sequence_log_1.loc['trace-02']['sequence'] == sequence_log_2.loc['trace-03']['sequence']
    # Check if single characters represent the same activities
    assert sequence_log_1.loc['trace-02']['sequence'][1] == sequence_log_2.loc['trace-03']['sequence'][1]
    assert sequence_log_1.loc['trace-02']['sequence'][2] == sequence_log_2.loc['trace-02']['sequence'][1]
    assert sequence_log_1.loc['trace-01']['sequence'][1] == sequence_log_2.loc['trace-03']['sequence'][3]


def test__event_log_to_activity_sequence():
    # Read event logs with similar timestamp distribution but different resources, activity names and trace IDs
    event_log = _read_event_log("./tests/assets/test_event_log_1.csv")
    mapping = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5', 'F': '6', 'G': '7', 'H': '8', 'I': '9'}
    # Check that the activity sequence transformation is done correctly
    ground_truth = pd.DataFrame(
        data={'sequence': ["12345689", "12435689", "12345789", "12435789"]},
        index=["trace-01", "trace-02", "trace-03", "trace-04"]
    )
    sequence_log = _event_log_to_activity_sequence(event_log, DEFAULT_CSV_IDS, mapping)
    assert ground_truth.equals(sequence_log)
    # Read event logs with similar timestamp distribution but different resources, activity names and trace IDs
    event_log = _read_event_log("./tests/assets/test_event_log_2.csv")
    mapping = {
        'Start': '1', 'Check stock': '2', 'Prepare package': '3',
        'Review packaging': '4', 'Prepare delivery': '5', 'Deliver route 1': '6',
        'Deliver route 2': '7', 'Delivery performed': '8', 'End': '9'
    }
    # Check that the activity sequence transformation is done correctly
    ground_truth = pd.DataFrame(
        data={'sequence': ["12345689", "12435689", "12345789", "12435789"]},
        index=["trace-10", "trace-11", "trace-12", "trace-13"]
    )
    sequence_log = _event_log_to_activity_sequence(event_log, DEFAULT_CSV_IDS, mapping)
    assert ground_truth.equals(sequence_log)
