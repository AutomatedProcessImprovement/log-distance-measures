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
# Call passing the event logs, its column ID mappings, timestamp type, and discretize function
import datetime

from log_similarity_metrics.config import DEFAULT_CSV_IDS
from log_similarity_metrics.cycle_time_emd import cycle_time_emd

emd = cycle_time_emd(
    event_log_1, DEFAULT_CSV_IDS,  # First event log and its column id mappings
    event_log_2, DEFAULT_CSV_IDS,  # Second event log and its column id mappings
    datetime.timedelta(hours=1)  # Bins of 1 hour
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
from log_similarity_metrics.config import AbsoluteHourEmdType, DEFAULT_CSV_IDS
from log_similarity_metrics.absolute_timestamps_emd import absolute_timestamps_emd, discretize_to_hour

# Call passing the event logs, its column ID mappings, timestamp type, and discretize function
emd = absolute_timestamps_emd(
    event_log_1, DEFAULT_CSV_IDS,  # First event log and its column id mappings
    event_log_2, DEFAULT_CSV_IDS,  # Second event log and its column id mappings
    AbsoluteHourEmdType.BOTH,  # Type of timestamp distribution (consider start times and/or end times)
    discretize_to_hour  # Function to discretize the absolute seconds of each timestamp (default by hour)
)
```

The timestamp EMD metric can be also used to compare the distribution of the start timestamps (with AbsoluteHourEmdType.START), or the end
timestamps (AbsoluteHourEmdType.END), instead of both of them.

Furthermore, the binning is performed to hour by default, but it can be customized passing another function discretize the total amount of
seconds to its bin.

```python
import math

from log_similarity_metrics.config import AbsoluteHourEmdType, DEFAULT_CSV_IDS
from log_similarity_metrics.absolute_timestamps_emd import absolute_timestamps_emd, discretize_to_day

# EMD of the (END) timestamps distribution where each bin represents a day
emd = absolute_timestamps_emd(
    event_log_1, DEFAULT_CSV_IDS,
    event_log_2, DEFAULT_CSV_IDS,
    AbsoluteHourEmdType.END,
    discretize_to_day
)

# EMD of the timestamps distribution where each bin represents a week
emd = absolute_timestamps_emd(
    event_log_1, DEFAULT_CSV_IDS,
    event_log_2, DEFAULT_CSV_IDS,
    discretize=lambda seconds: math.floor(seconds / 3600 / 24 / 7)
)
```

## Control-flow Log Similarity (CFLS)

Similarity measure between two event logs with the same number of traces (_L1_ and _L2_) comparing the control-flow dimension (see "Camargo
M, Dumas M, González-Rojas O. 2021. Discovering generative models from event logs: data-driven simulation vs deep learning. PeerJ Computer
Science 7:e577 https://doi.org/10.7717/peerj-cs.577" for a detailed description of the metric).

1. Transform each process trace of _L1_ and _L2_ to their corresponding activity sequence.
2. Compute the Damerau-Levenshtein distance between each trace _i_ from _L1_ and each trace _j_ of _L2_, and normalize it by dividing by the
   length of the longest trace.
3. Compute the matching between the traces of both logs (such that each _i_ is matched to a different _j_, and vice versa) minimizing the
   sum of distances with linear programming.
4. Transform the optimum distance values into similarity values by subtracting them to one (_1 - value_).
5. Compute the CFLS as the average of the normalized similarity values.

### Example of use

```python
from log_similarity_metrics.config import DEFAULT_CSV_IDS
from log_similarity_metrics.control_flow_log_similarity import control_flow_log_similarity

# Call passing the event logs, and its column ID mappings
emd = control_flow_log_similarity(
    event_log_1, DEFAULT_CSV_IDS,  # First event log and its column id mappings
    event_log_2, DEFAULT_CSV_IDS,  # Second event log and its column id mappings
)
```
