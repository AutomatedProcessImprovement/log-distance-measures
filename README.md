# Event Log Distance and Similarity Metrics

Python package with event log distance and similarity metrics.

### Example of input initialization

```python
import pandas as pd

from log_similarity_metrics.config import EventLogIDs

# Set event log column ID mapping
event_log_ids = EventLogIDs(  # This values are stored in DEFAULT_CSV_IDS
    case="case_id",
    activity="Activity",
    start_time="start_time",
    end_time="end_time",
    resource="Resource"
)
# Read and transform time attributes
event_log = pd.read_csv("/path/to/event_log.csv")
event_log[event_log_ids.start_time] = pd.to_datetime(event_log[event_log_ids.start_time], utc=True)
event_log[event_log_ids.end_time] = pd.to_datetime(event_log[event_log_ids.end_time], utc=True)
```

## Cycle Time EMD

Distance measure computing how different the cycle time discretized histograms of two event logs are.

1. Compute the cycle time of each process instance.
2. Group the cycle times in bins by a given bin size (time gap).
3. Compare the discretized histograms of the two event logs with the Wasserstein Distance (a.k.a. EMD).

### Example of use

```python
from log_similarity_metrics.config import AbsoluteHourEmdType, DEFAULT_CSV_IDS
from log_similarity_metrics.time import absolute_hour_emd, discretize_to_hour

# Call passing the event logs, its column ID mappings, timestamp type, and discretize function
emd = absolute_hour_emd(
    event_log_1, DEFAULT_CSV_IDS,  # First event log and its column id mappings
    event_log_2, DEFAULT_CSV_IDS,  # Second event log and its column id mappings
    AbsoluteHourEmdType.BOTH,  # Type of timestamp distribution (consider start times and/or end times)
    discretize_to_hour  # Function to discretize the absolute seconds of each timestamp (default by hour)
)
```

## Absolute Hour Timestamp EMD

Distance measure computing how different the histograms of the timestamps of two event logs are, discretizing the timestamps by absolute
hour.

1. Take all the start timestamps, the end timestamps, or both.
2. Group the timestamps by absolute hour (those timestamps between '02/05/2022 10:00:00' and '02/05/2022 10:59:59' goes to the same bin).
3. Compare the discretized histograms of the two event logs with the Wasserstein Distance (a.k.a. EMD).

### Example of use

```python
# Call passing the event logs, its column ID mappings, timestamp type, and discretize function
import datetime

from log_similarity_metrics.config import DEFAULT_CSV_IDS
from log_similarity_metrics.time import cycle_time_emd

emd = cycle_time_emd(
    event_log_1, DEFAULT_CSV_IDS,  # First event log and its column id mappings
    event_log_2, DEFAULT_CSV_IDS,  # Second event log and its column id mappings
    datetime.timedelta(hours=1)  # Bins of 1 hour
)
```