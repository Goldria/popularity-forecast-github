import csv
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


def parsingHTML(limit):
    url = "https://play.clickhouse.com/play?user=play"

    offset = 0
    data = []
    driver = webdriver.Chrome()

    while True:
        driver.get(url)

        query_input = driver.find_element(By.ID, 'query')

        select_query = f'SELECT \
                repo_name, \
                sum(event_type=\'ForkEvent\') AS forks, \
                sum(event_type=\'WatchEvent\') AS stars, \
                sum(event_type=\'IssuesEvent\') AS issues, \
                sum(event_type=\'CommitCommentEvent\') AS commits, \
                sum(event_type=\'PullRequestEvent\') AS pull_requests \
                FROM github_events WHERE(event_type IN(\'ForkEvent\', \'WatchEvent\', \'IssuesEvent\', \'CommitCommentEvent\', \'PullRequestEvent\')) \
                GROUP BY repo_name ORDER BY stars DESC, commits DESC LIMIT 2500 OFFSET {offset} \
                '
        query_input.send_keys(select_query)

        query_input.send_keys(Keys.CONTROL, Keys.RETURN)
        time.sleep(10)

        page_source = driver.page_source

        soup = BeautifulSoup(page_source, 'html.parser')

        table = soup.find('table', {'id': 'data-table'})

        for row in table.find_all('tr'):
            columns = row.find_all('td')[1:]
            row_data = [col.text.strip() for col in columns]
            data.append(row_data)

        offset += 2_500

        if offset < limit:
            break

    driver.quit()

    return data


data = parsingHTML(1_000_000)

filename = 'data_github.csv'
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
