import calendar
from os.path import isfile

import pandas as pd

from config import max_count
from utils import get_all_lines, pad_sentences, build_vocab

preprocessed_events_description = "data/barclays_events_description_preprocessed.csv"
assert isfile(preprocessed_events_description)
assert isfile("data/barclays_events.csv")
months = list(calendar.month_abbr)
df = pd.read_csv("data/barclays_events.csv", sep=", ")
preprocessed_descriptions = get_all_lines(preprocessed_events_description)
padded_description = pad_sentences(preprocessed_descriptions)
vocabulary, vocabulary_inv, word_counts = build_vocab(padded_description)
print("Length of vocab is: {}".format(len(vocabulary)))


def get_encoded_sentence(sentence):
    padded_sentece = [0] * max_count
    words = sentence.split(" ")
    for i in range(min(max_count, len(words))):
        padded_sentece[i] = vocabulary.get(words[i].strip(), 0)
    return padded_sentece


def get_all_events():
    events_info = {}
    with open(preprocessed_events_description) as f:
        desciptions = f.readlines()[1:]

    for i in range(df.shape[0]):
        month_abr, date_x, year = df["date"][i].split()
        date_x = int(date_x)
        if "TBA" in df["time"][i]:
            time = "07:00 PM"
        else:
            time = df["time"][i]
        date_string = '{}-{:02d}-{:02d} {}:00:00'.format(year, int(months.index(month_abr)), date_x, int(time.split(":")[0]) + 12)
        events_info[date_string] = {
            "name": df["event_title"][i],
            "description": desciptions[i],
            "padded_desciption": get_encoded_sentence(desciptions[i])
        }
        date_string = '{}-{:02d}-{:02d} {}:01:00'.format(year, int(months.index(month_abr)), date_x, int(time.split(":")[0]) + 12)
        events_info[date_string] = {
            "name": df["event_title"][i],
            "description": desciptions[i],
            "padded_desciption": get_encoded_sentence(desciptions[i])
        }
    return events_info
