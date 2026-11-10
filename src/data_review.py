import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your Kaggle datasets
reviews_df = pd.read_csv('data/steam_reviews_merged.csv')  
print("Loaded reviews data.")
games_df = pd.read_csv('data/steam_games.csv')      
print("Loaded games data.")
# Initial exploration
print(reviews_df.info())
print(reviews_df.head())
print(games_df.info())
print(games_df.head())

# Check for missing values
print(reviews_df.isnull().sum())
print(games_df.isnull().sum())

# Understand distributions
print(reviews_df['playtime_forever'].describe())
print(games_df['price'].describe())