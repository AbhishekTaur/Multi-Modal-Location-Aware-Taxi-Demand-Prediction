from collections import defaultdict
from os.path import join, isfile

import pandas as pd
from tabulate import tabulate

from config import p_zones, processed_taxi_data_root
from data_utils import compute_error, load_data, load_combined_data


def baseline_model():
    for pickup_zone in p_zones:
        assert isfile(join(processed_taxi_data_root, "{}_train.csv".format(pickup_zone))) and \
               isfile(join(processed_taxi_data_root, "{}_val.csv".format(pickup_zone))) and \
               isfile(join(processed_taxi_data_root, "{}_test.csv".format(pickup_zone))), \
            "Train, validation and test files does not exist, Please run download_and_preprocess_taxi_data.py"
        # metrics["split"] = ["Train", "Validation", "Test"]
        _, output = load_data(pickup_zone=pickup_zone, split="train")
        mean_value = int(output.mean())
        df = calculate_baseline_error(pickup_zone, mean_value)
        print("### Baseline model result for pickup zone: {}".format(pickup_zone))
        print(tabulate(df, tablefmt="pipe", headers="keys"), "\n")


def calculate_baseline_error(pickup_zone, predicted_value):
    metrics = defaultdict(list)
    _, output = load_data(pickup_zone=pickup_zone, split="train")
    train_error = compute_error(output, predicted_value)
    for k, v in train_error.items(): metrics[k].append(v)
    # print('baseline train error for {} is {}'.format(pickup_zone, train_error))
    _, output = load_data(pickup_zone=pickup_zone, split="val")
    validation_error = compute_error(output, predicted_value)
    for k, v in validation_error.items(): metrics[k].append(v)
    # print('baseline validation error for {} is {}'.format(pickup_zone, validation_error))
    _, output = load_data(pickup_zone=pickup_zone, split="test")
    test_error = compute_error(output, predicted_value)
    for k, v in test_error.items(): metrics[k].append(v)
    # print('baseline test error for {} is {}'.format(pickup_zone, test_error))
    df = pd.DataFrame(metrics, index=[[pickup_zone]*3])[:1]
    return df

if __name__ == '__main__':
    baseline_model()
