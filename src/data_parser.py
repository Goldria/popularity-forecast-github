import csv
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


class HTMLParser:
    def __init__(self):
        self.url = "https://play.clickhouse.com/play?user=play"
        self.offset = 0
        self.step = 1_000
        self.driver = webdriver.Chrome()

    def fetch_data(self, limit):
        data = []

        while True:
            self.driver.get(self.url)

            query_input = self.driver.find_element(By.ID, 'query')
            select_query = self.get_query()
            query_input.send_keys(select_query)
            query_input.send_keys(Keys.CONTROL, Keys.RETURN)
            time.sleep(15)

            page_source = self.driver.page_source
            if 'Exception: Not enough space on temporary disk' in page_source:
                continue

            soup = BeautifulSoup(page_source, 'html.parser')
            table = soup.find('table', {'id': 'data-table'})

            data.extend([
                [col.text.strip() for col in row.find_all('td')[1:]]
                for row in table.find_all('tr')
            ])

            self.offset += self.step

            if self.offset >= limit:
                break

        self.driver.quit()
        return data[:limit]

    def get_query(self):
        query = f"""
                SELECT
                    main.repo_name,
                    main.forks,
                    main.stars,
                    main.issues,
                    main.commits,
                    main.pull_requests,
                    cce.unique_commit_comment_authors AS commit_authors
                FROM (
                    SELECT
                        repo_name,
                        sum(event_type = 'ForkEvent') AS forks,
                        sum(event_type = 'WatchEvent') AS stars,
                        sum(event_type = 'IssuesEvent') AS issues,
                        sum(event_type = 'CommitCommentEvent') AS commits,
                        sum(event_type = 'PullRequestEvent') AS pull_requests,
                        COUNT(DISTINCT actor_login) AS unique_commit_authors
                    FROM
                        github_events
                    WHERE
                        event_type IN ('ForkEvent', 'WatchEvent', 'IssuesEvent', 'CommitCommentEvent', 'PullRequestEvent')
                        AND created_at >= DATE_SUB(CURDATE(), INTERVAL 8 YEAR)
                    GROUP BY
                        repo_name
                    HAVING
                        forks != 0 AND issues != 0 AND commits != 0 AND pull_requests != 0
                ) AS main
                LEFT JOIN (
                    SELECT
                        repo_name,
                        COUNT(DISTINCT actor_login) AS unique_commit_comment_authors
                    FROM
                        github_events
                    WHERE
                        event_type = 'CommitCommentEvent'
                        AND created_at >= DATE_SUB(CURDATE(), INTERVAL 8 YEAR)
                    GROUP BY
                        repo_name
                    ORDER BY
                        unique_commit_comment_authors DESC
                ) AS cce ON main.repo_name = cce.repo_name
                LIMIT {self.step} OFFSET {self.offset}
                """

        return query

    def save_to_csv(self, filename, limit):
        self.parsed_data = self.fetch_data(limit)
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.parsed_data)


filename = '../training_data/data_github.csv'

html_parser = HTMLParser()
parsed_data = html_parser.save_to_csv(filename, 300_000)
