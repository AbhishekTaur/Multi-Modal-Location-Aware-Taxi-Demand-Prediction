#!/usr/bin/env bash
export TF_CPP_MIN_LOG_LEVEL="3"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH=.
python train.py --location_aware False --model MLP --event_info False
python  train.py --location_aware True --model MLP --event_info False

python train.py --location_aware False --model LSTM --event_info False
python train.py --location_aware True --model LSTM --event_info False

python train.py --location_aware True --model MLP --event_info True
python  train.py --location_aware True --model LSTM --event_info True
