import os
import time
from os.path import isfile

from bs4 import BeautifulSoup
import urllib.request

user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
headers = {'User-Agent': user_agent, }

os.makedirs("./data/", exist_ok=True)


def get_event_description(url):
    request = urllib.request.Request(url, None, headers)  # The assembled request
    response = urllib.request.urlopen(request)
    soup = BeautifulSoup(response.read(), 'html.parser')
    description = ""
    for p_desc in soup.findAll("div", {"class": "descriptionContainer"})[0].findAll("p"):
        description += p_desc.get_text().replace('\n', ' ').replace('\r', '').strip() + " "
    return description


print("Scraping event description from {}".format("barclayscenter.com"))

events_csv = "./data/barclays_events.csv"
assert isfile(events_csv), "Events file does not exists. Please run fetch_list_of_events.py"

with open(events_csv, "r") as f:
    data = f.readlines()[1:]

with open('./data/barclays_events_description.csv', 'w') as f:
    print("description", file=f)
    for line in data:
        date, time_s, event_title, link = line.split(", ")
        print("Fetching description for event: {} {}".format(date, event_title))
        print("{}".format(get_event_description(link)), file=f)
        time.sleep(3)
