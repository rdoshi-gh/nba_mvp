import os
import pandas as pd
from bs4 import BeautifulSoup
import requests

def get_season_data(year):
    #Get data
    url = f'https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html' 
    res = requests.get(url)
    #Parse data
    soup = BeautifulSoup(res.text, 'html.parser')
    table = soup.find('table', {'id': 'per_game_stats'})
    #Read data
    df = pd.read_html(str(table))[0]
    df = df[df['Player'] != 'Player']
    df['Season'] = f'{year-1}-{str(year)[-2:]}'
    return df

def collect_historical_stats(start=2000, end=2025):
    all_seasons = []
    for year in range(start, end + 1):
        print(f'Fetching {year}...')
        df = get_season_data(year)
        all_seasons.append(df)
    full_df = pd.concat(all_seasons, ignore_index=True)
    os.makedirs('data/raw', exist_ok=True)
    full_df.to_csv('data/raw/player_stats_raw.csv', index=False)

if __name__ == "__main__":
    collect_historical_stats()
