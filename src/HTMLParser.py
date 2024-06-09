import csv
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


class HTMLParser:
    """
    HTMLParser - class to parse HTML data from a specified URL using Selenium WebDriver and BeautifulSoup, 
    and fetch data from an HTML table.

    Attributes:
        url (str): the URL from which data is fetched.
        offset (int): the offset value used for pagination.
        step (int): the step value used for pagination.
        driver (object): an instance of the Selenium WebDriver for Chrome.

    Methods:
        fetch_data(limit):
            Fetches data from the HTML table on the webpage.

        get_query():
            Generates a SQL query to retrieve data from the database.


        save_to_csv(filename, limit):
            Fetches data and saves it to a CSV file.
    """

    def __init__(self):
        """
        Initializes the HTMLParser object.
        """
        self.url = "https://play.clickhouse.com/play?user=play"
        self.offset = 0
        self.step = 1_000
        self.driver = webdriver.Chrome()

    def fetch_data(self, limit):
        """
        Fetches data from the HTML table on the webpage.

        Args:
            limit (int): The maximum number of rows to fetch.

        Returns:
            List of lists containing the fetched data.
        """
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
        """
        Generates a SQL query to retrieve data from the database.

        Returns:
            SQL query string.
        """
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
        """
        Fetches data and saves it to a CSV file.

        Args:
            filename (str): The name of the CSV file to save the data to.
            limit (int): The maximum number of rows to fetch and save.

        Returns:
            None.
        """
        self.parsed_data = self.fetch_data(limit)
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.parsed_data)
