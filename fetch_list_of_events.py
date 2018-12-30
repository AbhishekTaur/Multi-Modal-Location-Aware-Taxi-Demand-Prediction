import os
import time

from bs4 import BeautifulSoup
from selenium import webdriver

os.makedirs("./data/", exist_ok=True)

with webdriver.Firefox(executable_path='./geckodriver') as driver:
    driver.implicitly_wait(30)
    driver.maximize_window()
    driver.get("https://www.barclayscenter.com/events/event-calendar")
    time.sleep(5)
    print("Scraping data from {}".format("barclayscenter.com"))
    with open('./data/barclays_events.csv', 'w') as f:
        print("date, time, event_title, link", file=f)
        for _ in range(0, 23):
            html_source = driver.page_source
            soup = BeautifulSoup(html_source, 'html.parser')
            month, year = soup.find(id='cal-month').get_text().split()
            print("Fetching events for {} {}".format(month, year))
            for element in soup.findAll("div", {"class": "fc-content"})[::-1]:
                date = element.findAll("span", {"class": "dt"})[0].get_text().replace(", ", " ").replace("-", "").strip()
                time_s = element.findAll("span", {"class": "time"})[0].get_text().strip()
                link = element.findAll("a")[0].get("href")
                event_title = element.findAll("a")[0].get_text().strip().replace(", ", " ").replace(",", " ")
                print("{}, {}, {}, {}".format(date, time_s, event_title, link), file=f)
            driver.find_element_by_xpath(".//*[@id='cal-prev']").click()
            time.sleep(2)