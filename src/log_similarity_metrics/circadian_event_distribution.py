from statistics import mean

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from log_similarity_metrics.config import EventLogIDs, AbsoluteTimestampType


def circadian_event_distribution_distance(
        event_log_1: pd.DataFrame,
        log_1_ids: EventLogIDs,
        event_log_2: pd.DataFrame,
        log_2_ids: EventLogIDs,
        discretize_type: AbsoluteTimestampType = AbsoluteTimestampType.BOTH,
        normalize: bool = True
) -> float:
    """
    EMD (or Wasserstein Distance) between the distribution of timestamps of two event logs, windowed by weekday (e.g. the instants
    happening on all Mondays are compared together), and discretized to their hour.

    :param event_log_1: first event log.
    :param log_1_ids: mapping for the column IDs of the first event log.
    :param event_log_2: second event log.
    :param log_2_ids: mapping for the column IDs for the second event log.
    :param discretize_type: type of EMD measure (only take into account start timestamps, only end timestamps, or both).
    :param normalize: whether to normalize the distance metric to a value in [0.0, 1.0]

    :return: the EMD between the timestamp distribution of the two event logs windowed by weekday.
    """
    # Get discretized start and/or end timestamps
    discretized_instants_1 = _discretize(event_log_1, log_1_ids, discretize_type)
    discretized_instants_2 = _discretize(event_log_2, log_2_ids, discretize_type)
    # Get unique weekdays
    weekdays = set(
        np.append(discretized_instants_1['weekday'].unique(), discretized_instants_2['weekday'].unique())
    )
    # Compute the distance between the instant in the event logs for each weekday
    distances = []
    for week_day in weekdays:
        window_1 = discretized_instants_1[discretized_instants_1['weekday'] == week_day]['hour']
        window_2 = discretized_instants_2[discretized_instants_2['weekday'] == week_day]['hour']
        if len(window_1) > 0 and len(window_2) > 0:
            distances += [wasserstein_distance(window_1, window_2)]
        else:
            distances += [23]  # 23 is the maximum EMD value for two histograms with values between 0 and 23.
    # Compute distance metric
    distance = mean(distances)
    if normalize:
        distance = distance / 23
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
