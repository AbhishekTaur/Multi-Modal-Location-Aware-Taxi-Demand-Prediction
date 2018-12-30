import datetime
import os.path
from collections import defaultdict
from functools import lru_cache

import pandas as pd
import wget
import numpy as np
from tabulate import tabulate

from config import p_zones, years, months, processed_taxi_data_root, max_count
from preprocess_event_info import get_all_events

count_dict = {}


def get_ordered_time(start_date, end_date):
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]
    for date in date_generated:
        for hour in range(0, 24):
            for minute in [0, 1]:
                yield "{}".format(date + datetime.timedelta(hours=hour) + datetime.timedelta(minutes=minute))


def train_test_split():
    for pickupzone in p_zones:
        filename = "{}.csv".format(pickupzone)
        filepath = os.path.join(processed_taxi_data_root, filename)
        df = pd.read_csv(filepath)
        print("\n", "### Data summary for pickup zone: {}".format(pickupzone))
        print(tabulate(df.describe(), tablefmt="pipe", headers="keys"))
        total_data_points = df.count(axis='rows')['year']
        df_train = df[:int(total_data_points * 0.7)]
        df_val = df[int(total_data_points * 0.7):int(total_data_points * 0.8)]
        df_test = df[int(total_data_points * 0.8):]
        test_file = os.path.join(processed_taxi_data_root, filename.split('.')[0] + '_test.csv')
        train_file = os.path.join(processed_taxi_data_root, filename.split('.')[0] + '_train.csv')
        validation_file = os.path.join(processed_taxi_data_root, filename.split('.')[0] + '_val.csv')
        df_test.to_csv(test_file, index=False)
        df_train.to_csv(train_file, index=False)
        df_val.to_csv(validation_file, index=False)


@lru_cache(maxsize=2000)
def get_weekday(year, month, day):
    return datetime.date(int(year), int(month), int(day)).isoweekday()


def download_and_preprocess_data():
    events_info = get_all_events()
    events_saved =False
    events = []
    preprocessed = []
    for pickup_zone in p_zones:
        count_dict[pickup_zone] = defaultdict(int)
    for year in years:
        for month in months:
            if year == '2018' and month == '07':
                break
            filename = 'green_tripdata_' + year + '-' + month + '.csv'
            filepath = os.path.join('/home/mgo/taxi_data', filename)
            if not os.path.isfile(filepath):
                print("Downloading data for {} {}".format(year, month))
                url = 'https://s3.amazonaws.com/nyc-tlc/trip data/green_tripdata_{0}-{1}.csv'.format(year, month)
                wget.download(url, out='/home/mgo/taxi_data')

            df = pd.read_csv(filepath)
            df = df[df["PULocationID"].isin(p_zones)]
            print("Preprocessing data for {} {}".format(year, month))
            for date_time, pickup_zone in zip(df['lpep_pickup_datetime'], df['PULocationID']):
                date, time_s = date_time.split(" ")
                hour, minute, _ = time_s.split(':')
                minute = "01" if (int(minute) > 30) else "00"
                time_stamp = date + '-' + hour + '-' + minute
                count_dict[pickup_zone][time_stamp] += 1

            start_date = year + '-' + month + '-' + '01'

            if int(month) == 12:
                next_month = '01'
                next_year = str(int(year) + 1)
                end_date = next_year + '-' + next_month + '-' + '01'
            else:
                next_month = "{:02d}".format(int(month) + 1)
                end_date = year + '-' + next_month + '-' + '01'

            if next_month != '--':
                for pickup_zone in p_zones:
                    processed_file = os.path.join(processed_taxi_data_root, "{}.csv".format(pickup_zone))
                    if pickup_zone not in preprocessed:
                        preprocessed.append(pickup_zone)
                        with open(processed_file, "w") as f:
                            print("year,month,day,hour,minute,weekday,barclays_event,pickup_no", file=f)

                    with open(processed_file, "a") as f:
                        for ordered_time in get_ordered_time(start_date, end_date):
                            if not events_saved:
                                if pickup_zone == p_zones[0]:
                                    events.append(events_info.get(ordered_time, {"padded_desciption":[0]*max_count})["padded_desciption"])
                            date, time = ordered_time.split(" ")
                            day = date.split('-')[-1]
                            hour, minute, _ = time.split(":")
                            formated_time = date + '-' + hour + '-' + minute
                            print("{},{},{},{},{},{},{},{}".format(year, month, day, hour, minute, get_weekday(year, month, day), 1 if sum(events_info.get(ordered_time, {"padded_desciption":[0]*max_count})["padded_desciption"])!=0 else 0,  count_dict[pickup_zone][formated_time]),
                                  file=f)
    events_arr = np.array(events)
    print(events_arr.shape)
    np.save(os.path.join(processed_taxi_data_root, "events"), events_arr)


def main():
    os.makedirs('taxi_data', exist_ok=True)
    os.makedirs(processed_taxi_data_root, exist_ok=True)
    download_and_preprocess_data()
    train_test_split()


if __name__ == '__main__':
    main()
