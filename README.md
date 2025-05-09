# nba_mvp
Predicts the season's NBA MVP

How to use: Run scrape_players.py to get the player data, the data will be stored in a raw data csv file. Next, run label_mvp.py to classify players on whether or not they have been named MVP in the past. Then, run process_data.py to get rid of useless values and predict results on the set using the given features, which will store the data in a new csv. Lastly, run predict_mvp.py to get the top 10 most likely candidates for the MVP in the given season based on the given features.
