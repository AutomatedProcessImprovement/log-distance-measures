from statistics import mean

import pandas as pd
from scipy.stats import wasserstein_distance

from log_distance_measures.config import EventLogIDs, AbsoluteTimestampType, DistanceMetric
from log_distance_measures.earth_movers_distance import earth_movers_distance


def circadian_event_distribution_distance(
        original_log: pd.DataFrame,
        original_ids: EventLogIDs,
        simulated_log: pd.DataFrame,
        simulated_ids: EventLogIDs,
        discretize_type: AbsoluteTimestampType = AbsoluteTimestampType.BOTH,
        metric: DistanceMetric = DistanceMetric.WASSERSTEIN,
        normalize: bool = False
) -> float:
    """
    EMD (or Wasserstein Distance) between the distribution of timestamps of two event logs, windowed by weekday (e.g. the instants
    happening on all Mondays are compared together), and discretized to their hour.

    :param original_log: first event log.
    :param original_ids: mapping for the column IDs of the first event log.
    :param simulated_log: second event log.
    :param simulated_ids: mapping for the column IDs for the second event log.
    :param discretize_type: type of EMD measure (only take into account start timestamps, only end timestamps, or both).
    :param metric: distance metric to use in the histogram comparison.
    :param normalize: whether to normalize the distance metric to a value in [0.0, 1.0]

    :return: the EMD between the timestamp distribution of the two event logs windowed by weekday.
    """
    # Get discretized start and/or end timestamps
    original_discrete_events = _discretize(original_log, original_ids, discretize_type)
    simulated_discrete_events = _discretize(simulated_log, simulated_ids, discretize_type)
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
        discretize_type: AbsoluteTimestampType = AbsoluteTimestampType.BOTH
) -> pd.DataFrame:
    """
        Create a pd.Dataframe with the hour (0-23) of the timestamps (start, end, or both, depending on [discretize_type]) of the events
        in log [event_log] in one column ('hour'), and the day of the week in another column ('weekday').

        :param event_log: event log to extract the instants of.
        :param log_ids: mapping for the column IDs of the event log.
        :param discretize_type: type of EMD measure (only take into account start timestamps, only end timestamps, or both).

        :return: A pd.Dataframe with the hour of each discretized instant in the one column, and the weekday in other.
        """
    # Get the instants to discretize
    if discretize_type == AbsoluteTimestampType.BOTH:
        to_discretize = pd.concat(
            [event_log[log_ids.start_time], event_log[log_ids.end_time]]
        ).reset_index(drop=True).to_frame(name='instant')
    elif discretize_type == AbsoluteTimestampType.START:
        to_discretize = event_log[log_ids.start_time].to_frame(name='instant')
    else:
        to_discretize = event_log[log_ids.end_time].to_frame(name='instant')
    # Compute their weekday
    to_discretize['weekday'] = to_discretize['instant'].apply(lambda instant: instant.day_of_week)
    to_discretize['hour'] = to_discretize['instant'].apply(lambda instant: instant.hour)
    to_discretize.drop(['instant'], axis=1, inplace=True)
    # Return discretized timestamps
    return to_discretize
