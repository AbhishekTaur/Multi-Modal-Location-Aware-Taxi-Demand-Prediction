import argparse
import os
import warnings
from collections import defaultdict
from os.path import join

warnings.simplefilter(action='ignore', category=FutureWarning)

from tabulate import tabulate
from tensorflow.python.keras.callbacks import ModelCheckpoint
import pandas as pd

from config import p_zones, exp_logs_root, pzone_encoder
from data_utils import load_data, compute_error, str2bool, load_combined_data
from model_utils import build_model

os.makedirs(exp_logs_root, exist_ok=True)

parser = argparse.ArgumentParser(description='Audio Visual Mask Estimation for Speech Separation')
parser.add_argument("--date_time", type=str2bool, default=False)
parser.add_argument("--event_info", type=str2bool, default=False)
parser.add_argument("--location_aware", type=str2bool, default=False)
parser.add_argument('--prefix', type=str, help='Prefix for experiments artifacts')
parser.add_argument('--model', type=str, default="MLP", help='Type of model: MLP / LSTM')
parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of iterations for training')
parser.add_argument('--lags', type=int, default=48)
args = parser.parse_args()

event_info = args.event_info
date_time = args.date_time
model_type = args.model
lags = args.lags
metrics = defaultdict(list)
model = build_model(lags, model=model_type, summary=True, date_time=args.date_time, combined_model=args.location_aware, event_info=event_info)

if args.location_aware:
    X_train, y_train = load_combined_data(lags=lags, split="train", date_time=date_time, event_info=event_info)
    X_val, y_val = load_combined_data(lags=lags, split="val", date_time=date_time, event_info=event_info)
    weights_model = join(exp_logs_root, "{}_{}_{}_weights.best.hdf5".format(lags, model_type, "combined"))
    model_checkpoint = ModelCheckpoint(weights_model, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    model.fit(X_train, y_train, epochs=args.epochs, verbose=1,  batch_size=args.batch_size, validation_data=(X_val, y_val),
              callbacks=[model_checkpoint])
    model.load_weights(weights_model)
    for pickup_zone in p_zones:
        X_test, y_test = load_data(pickup_zone, lags=lags, split="test", date_time=date_time, event_info=event_info)
        X_test["pickup_zone"] = pzone_encoder.transform([pickup_zone] * y_test.shape[0])
        preds = model.predict(X_test)
        test_error = compute_error(y_test, preds)
        for k, v in test_error.items(): metrics[k].append(v)
else:
    for pickup_zone in p_zones:
        weights_model = join(exp_logs_root, "{}_{}_{}_weights.best.hdf5".format(lags, model_type, pickup_zone))
        model_checkpoint = ModelCheckpoint(weights_model, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
        X_train, y_train = load_data(pickup_zone, lags=lags, split="train", date_time=date_time, event_info=event_info)
        X_val, y_val = load_data(pickup_zone, lags=lags, split="val", date_time=date_time, event_info=event_info)
        X_test, y_test = load_data(pickup_zone, lags=lags, split="test", date_time=date_time, event_info=event_info)
        model.fit(X_train, y_train, epochs=args.epochs, verbose=1,  batch_size=args.batch_size, validation_data=(X_val, y_val),
                  callbacks=[model_checkpoint])
        model.load_weights(weights_model)
        preds = model.predict(X_test)
        test_error = compute_error(y_test, preds)
        for k, v in test_error.items(): metrics[k].append(v)
        model = build_model(lags, model=model_type, summary=True, date_time=args.date_time, combined_model=args.location_aware, event_info=event_info)
        os.remove(weights_model)
        print(metrics)

df = pd.DataFrame(metrics, index=p_zones)
print(tabulate(df, tablefmt="pipe", headers="keys"), "\n")
