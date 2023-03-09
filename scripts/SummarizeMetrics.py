import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import t


###################################
# - COMPUTE CONFIDENCE INTERVAL - #
###################################

def compute_mean_conf_interval(data: list, confidence: float = 0.95) -> Tuple[float, float]:
    # Compute the sample mean and standard deviation
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)  # ddof=1 calculates the sample standard deviation
    # Compute the degrees of freedom
    df = len(data) - 1
    # Compute the t-value for the confidence level
    t_value = t.ppf(1 - (1 - confidence) / 2, df)
    # Compute the standard error of the mean
    std_error = sample_std / np.sqrt(len(data))
    conf_interval = t_value * std_error
    # Compute the confidence interval
    return sample_mean, conf_interval


###################
# - MAIN SCRIPT - #
###################

if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser(description="Summarize a set of metrics reporting the mean and confidence interval.")
    parser.add_argument("csv_file", help="CSV file with the metrics to summarize")
    args = parser.parse_args()
    # Read csv file
    results = pd.read_csv(args.csv_file)
    results.drop('name', inplace=True)
    # Process each column
    summary = results.apply(lambda col: compute_mean_conf_interval(col.to_list())).set_axis(['mean', 'conf'])
    # Output data
    summary.to_csv(args.csv_file[:args.csv_file.rfind('.')] + "_summary.csv")
