from statistics import mean

import pandas as pd
from scipy.stats import wasserstein_distance

from log_distance_measures.config import EventLogIDs, DistanceMetric
from log_distance_measures.earth_movers_distance import earth_movers_distance


def circadian_workforce_distribution_distance(
        original_log: pd.DataFrame,
        original_ids: EventLogIDs,
        simulated_log: pd.DataFrame,
        simulated_ids: EventLogIDs,
        metric: DistanceMetric = DistanceMetric.WASSERSTEIN,
        normalize: bool = False
) -> float:
    """
    EMD (or Wasserstein Distance) between the distribution of the workforce by hour (number of observed active workers
    in each hour) in two event logs, windowed by weekday (e.g., the workforces measured for each Monday in a log are
    aggregated as an average into one single Monday).

    :param original_log: first event log.
    :param original_ids: mapping for the column IDs of the first event log.
    :param simulated_log: second event log.
    :param simulated_ids: mapping for the column IDs for the second event log.
    :param metric: distance metric to use in the histogram comparison.
    :param normalize: whether to normalize the distance metric to a value in [0.0, 1.0]

    :return: the EMD between the timestamp distribution of the two event logs windowed by weekday.
    """
    # Get discretized start and/or end timestamps
    original_discrete_events = _discretize(original_log, original_ids)
    simulated_discrete_events = _discretize(simulated_log, simulated_ids)
    # Compute the distance between the instant in the event logs for each weekday
    distances = []
    for week_day in range(7):  # All weekdays
        original_window = original_discrete_events[original_discrete_events['weekday'] == week_day]['hour']
        simulated_window = simulated_discrete_events[simulated_discrete_events['weekday'] == week_day]['hour']
        if len(original_window) > 0 and len(simulated_window) > 0:
            # Both have observations in this weekday
            if metric == DistanceMetric.EMD:
                distances += [earth_movers_distance(original_window, simulated_window) / len(original_window)]
            else:
                distances += [wasserstein_distance(original_window, simulated_window)]
        elif len(original_window) == 0 and len(simulated_window) == 0:
            # Both have no observations in this weekday
            distances += [0.0]
        else:
            # Only one has observations in this weekday, penalize with max distance value
            distances += [23.0]  # 23 is the maximum value for two histograms with values between 0 and 23.
    # Compute distance metric
    distance = mean(distances)
    if normalize:
        distance = distance / 23.0
    # Return metric
    return distance


def _discretize(
        event_log: pd.DataFrame,
        log_ids: EventLogIDs,
) -> pd.DataFrame:
    """
    Create a pd.Dataframe storing, for each hour (0-23) of each weekday, the number of different resources
    that were actively working (an event associated to them was registered in that hour) through the entire event log.

    :param event_log: event log to measure the workforces of.
    :param log_ids: mapping for the column IDs of the event log.

    :return: A pd.Dataframe with the active workforce for each hour for each day of the week.
    """
    # Get the instants to discretize
    start_times = event_log[[log_ids.start_time, log_ids.resource]].rename(columns={log_ids.start_time: 'instant'})
    end_times = event_log[[log_ids.end_time, log_ids.resource]].rename(columns={log_ids.end_time: 'instant'})
    to_discretize = pd.concat([start_times, end_times]).reset_index(drop=True)
    # Compute their weekday
    to_discretize['weekday'] = to_discretize['instant'].apply(lambda instant: instant.day_of_week)
    to_discretize['hour'] = to_discretize['instant'].apply(lambda instant: instant.hour)
    to_discretize['day-hour'] = to_discretize['instant'].apply(lambda instant: instant.strftime(format="%Y-%m-%d %H"))
    # Keep unique resources
    to_discretize.drop_duplicates(subset=['day-hour', log_ids.resource], inplace=True)
    to_discretize.drop(['instant', 'day-hour', log_ids.resource], axis=1, inplace=True)
    # Compute number of unique resources per weekday & hour
    # workforce = to_discretize.groupby(['weekday', 'hour']).size().reset_index().rename(columns={0: 'workforce'})
    return to_discretize
