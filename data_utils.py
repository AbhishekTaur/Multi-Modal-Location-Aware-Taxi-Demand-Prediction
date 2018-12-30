import argparse
from collections import OrderedDict, defaultdict
from os.path import join, isfile

import numpy as np
import pandas as pd

from config import p_zones, processed_taxi_data_root, encoder, pzone_encoder


def str2bool(v: str):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_data(pickup_zone, lags=1, split="train", date_time=False, event_info=False):
    assert split in ["train", "test", "val"], "split should be one of train, test and val"
    assert isfile(join(processed_taxi_data_root, "{}_train.csv".format(pickup_zone))) and \
           isfile(join(processed_taxi_data_root, "{}_val.csv".format(pickup_zone))) and \
           isfile(join(processed_taxi_data_root, "{}_test.csv".format(pickup_zone))), \
        "Train, validation and test files does not exist, Please run download_and_preprocess_taxi_data.py"
    assert pickup_zone in p_zones, "pickup zone must be in p_zones"
    inputs = {}
    df = pd.read_csv(join(processed_taxi_data_root, "{}_{}.csv".format(pickup_zone, split)))
    df_predicted_pickups = df["pickup_no"][1:]

    df2 = df["pickup_no"]
    df_pickup_lags = pd.concat([pd.Series(df2.shift(x)) for x in range(lags)[::-1]], axis=1).fillna(0)[:-1]
    df_pickup_lags = df_pickup_lags.values
    inputs["pickup_lags"] = df_pickup_lags
    if date_time:
        date_time_feats = [encoder[feat].transform(df[feat][:-1].tolist()) for feat in encoder.keys()]
        date_time = np.concatenate(date_time_feats, axis=1)
        inputs["date_time"] = date_time
    if event_info:
        # inputs["events"] = np.repeat(df["barclays_event"].values[:-1], 2).reshape(-1, 2)
        events = np.load("processed_taxi_data/events.npy")
        if split == "train":
            start_index = 0
            end_index = int(0.7 * events.shape[0])
        elif split == "val":
            end_index = int(0.8 * events.shape[0])
            start_index = int(0.7 * events.shape[0])
        else:
            start_index = int(0.8 * events.shape[0])
            end_index = events.shape[0]
        inputs["events"] = events[start_index:end_index-1]
        print(events.shape, start_index, end_index, df_predicted_pickups.values.shape)
    return inputs, df_predicted_pickups.values


def load_combined_data(**kwargs):
    inputs = defaultdict(list)
    outputs = []
    for pickup_zone in p_zones:
        inp, out = load_data(pickup_zone, **kwargs)
        for k, v in inp.items(): inputs[k].append(v)
        inputs["pickup_zone"].append(pzone_encoder.transform([pickup_zone] * out.shape[0]))
        outputs.append(out)
    for k, v in inputs.items(): inputs[k] = np.concatenate(inputs[k], axis=0)
    outputs = np.concatenate(outputs, axis=0)
    return inputs, outputs


def compute_error(trues, predicted):
    mae = np.mean(np.abs(predicted - trues))
    rmse = np.sqrt(np.mean((predicted - trues) ** 2))
    return OrderedDict(mae=mae, rmse=rmse)


if __name__ == '__main__':
    load_data(p_zones[0], lags=5)
