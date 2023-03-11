"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup
import csv
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

"""
#Website: https://fantasy.espn.com/football/leaders
"""

# Set the URL of the page to scrape
url = 'https://fantasy.espn.com/football/leaders'

# Make a GET request to the URL
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Find all the div elements with class "flex"
divs = soup.find_all('div', class_='flex')

print(divs)

# Open a CSV file to write the data
with open('fantasy_football_data.csv', mode='w') as csv_file:
    fieldnames = ['Player', 'Position', 'Team', 'Projected Points']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Loop through each div element
    for div in divs:
        # Find the data elements within the div
        player_elem = div.find('span', class_='truncate')
        position_elem = div.find('span', class_='playerinfo_playerteam')
        team_elem = div.find('span', class_='playerinfo_playerpos ttu')
        projected_points_elem = div.find('div', class_='fw-medium ml2')

        print(player_elem)
        print(position_elem)
        print(team_elem)
        print(projected_points_elem)

        # Extract the data from the elements if they exist
        player = player_elem.get_text().strip() if player_elem else None
        position = position_elem.get_text().strip() if position_elem else None
        team = team_elem.get_text().strip() if team_elem else None
        projected_points = projected_points_elem.get_text().strip() if projected_points_elem else None

        # Write the data to the CSV file
        writer.writerow({'Player': player, 'Position': position, 'Team': team, 'Projected Points': projected_points})
"""
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
import time
import csv

# Set the URL of the page to scrape
url = 'https://fantasy.espn.com/football/leaders'

# Set up the Selenium web driver
edge_options = Options()
edge_options.add_argument('--headless')
driver = webdriver.Edge(options=edge_options)

# Load the web page using Selenium
driver.get(url)

# Wait for the page to fully load
time.sleep(5)

# Find the table element on the page
table = driver.find_element(By.CLASS_NAME, 'Table__TBODY')

print("Table: " + str(table))

# Find all the rows within the table
rows = table.find_elements(By.CLASS_NAME, 'tr')

print("Rows: " + str(rows))

# Open a CSV file to write the data
with open('fantasy_football_data.csv', mode='w') as csv_file:
    fieldnames = ['Player', 'Position', 'Team', 'Projected Points']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Loop through each row in the table
    for row in rows:
        # Find all the columns within the row
        columns = row.find_elements(By.CLASS_NAME, 'td')
        player_elem = columns[0].find_element(By.CLASS_NAME, 'AnchorLink')
        position_elem = columns[1]
        team_elem = columns[2]
        projected_points_elem = columns[3]

        """print(player_elem.text.strip())
        print(position_elem.text.strip())
        print(team_elem.text.strip())
        print(projected_points_elem.text.strip())"""

        # Extract the data from the elements
        player = player_elem.text.strip()
        position = position_elem.text.strip()
        team = team_elem.text.strip()
        projected_points = projected_points_elem.text.strip()

        # Write the data to the CSV file
        writer.writerow({'Player': player, 'Position': position, 'Team': team, 'Projected Points': projected_points})

# Close the Selenium web driver
driver.quit()
