import enum
import math
from dataclasses import dataclass


@dataclass
class EventLogIDs:
    case: str = 'case'
    activity: str = 'activity'
    start_time: str = 'start_timestamp'
    end_time: str = 'end_timestamp'


DEFAULT_CSV_IDS = EventLogIDs(case='case_id',
                              activity='Activity',
                              start_time='start_time',
                              end_time='end_time')
DEFAULT_XES_IDS = EventLogIDs(case='case:concept:name',
                              activity='concept:name',
                              start_time='time:start',
                              end_time='time:timestamp')


class AbsoluteTimestampType(enum.Enum):
    BOTH = 0
    START = 1
    END = 2


class DistanceMetric(enum.Enum):
    EMD = 0
    WASSERSTEIN = 1


def discretize_to_minute(seconds: int):
    return math.floor(seconds / 60)


def discretize_to_hour(seconds: int):
    return math.floor(seconds / 3600)


def discretize_to_day(seconds: int):
    return math.floor(seconds / 3600 / 24)
