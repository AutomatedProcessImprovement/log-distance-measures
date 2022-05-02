# Event Log Distance and Similarity Metrics

Python package with event log distance and similarity metrics.

## Cycle Time EMD

Distance measure computing how different the cycle time discretized histograms of two event logs are.

1. Compute the cycle time of each process instance.
2. Group the cycle times in bins by a given bin size (time gap).
3. Compare the discretized histograms of the two event logs with the Wasserstein Distance (a.k.a. EMD).

## Absolute Hour Timestamp EMD

Distance measure computing how different the histograms of the timestamps of two event logs are, discretizing the timestamps by absolute hour.

1. Take all the start timestamps, the end timestamps, or both.
2. Group the timestamps by absolute hour (those timestamps between '02/05/2022 10:00:00' and '02/05/2022 10:59:59' goes to the same bin).
3. Compare the discretized histograms of the two event logs with the Wasserstein Distance (a.k.a. EMD).
