import enum
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


class AbsoluteHourEmdType(enum.Enum):
    BOTH = 0
    START = 1
    END = 2
