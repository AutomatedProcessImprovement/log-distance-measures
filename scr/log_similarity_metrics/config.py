from dataclasses import dataclass


@dataclass
class EventLogIDs:
    case: str = 'case'
    activity: str = 'activity'
    start_time: str = 'start_timestamp'
    end_time: str = 'end_timestamp'
    resource: str = 'resource'
