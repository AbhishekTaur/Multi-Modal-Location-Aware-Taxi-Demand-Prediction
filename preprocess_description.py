import os
import statistics
from os.path import isfile, join

from utils import clean_text, get_all_lines, pad_sentences, build_vocab
import matplotlib.pyplot as plt
import numpy as np

num_of_words = []
events_description = "data/barclays_events_description.csv"
preprocessed_events_description = "data/barclays_events_description_preprocessed.csv"
assert isfile(events_description), "Event description does not exists. Please run fetch_event_description.py"
exp_logs_root = "./exp_logs/"
os.makedirs(exp_logs_root, exist_ok=True)

descriptions = get_all_lines(events_description)

with open(preprocessed_events_description, "w") as f:
    for description in descriptions:
        preprocessed_desc = clean_text(description)
        num_of_words.append(preprocessed_desc.count(" ") + 1)
        print(preprocessed_desc, file=f)

plt.hist(num_of_words, bins='auto')
plt.xlabel("Number of words in the description")
plt.ylabel("Frequency")
plt.savefig(join(exp_logs_root, "desc_text_hist.pdf"), dpi=300, format="pdf")

median = statistics.median_high(num_of_words)
print("Median length is: {}".format(median))

