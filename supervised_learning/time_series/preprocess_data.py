#!/usr/bin/env python3
"""preprocess data"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# load data
coinbase_df = pd.read_csv(
    'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')
bitstamp_df = pd.read_csv(
    'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')
# show first 5 rows
print(coinbase_df.head())
print(bitstamp_df.head())
# delete null values rows
coinbase_df.dropna(inplace=True)
bitstamp_df.dropna(inplace=True)
# check for missing values
print("number of missing values")
print(coinbase_df.isnull().sum())
print(bitstamp_df.isnull().sum())
# show first 5 rows after removing null values
print('')
print('data without null values\n')
print(coinbase_df.head())
print(bitstamp_df.head())
# show stats of data
print(coinbase_df.describe())
print(bitstamp_df.describe())
# choose important features
columns_to_keep = [
    'Open',
    'High',
    'Low',
    'Close',
    'Volume_(BTC)',
    'Volume_(Currency)',
    'Timestamp']
coinbase_df = coinbase_df[columns_to_keep]
bitstamp_df = bitstamp_df[columns_to_keep]
# merge data
combined_df = pd.concat([coinbase_df, bitstamp_df], axis=0, ignore_index=True)

# sort by Timestamp
combined_df.sort_values('Timestamp', inplace=True)
combined_df.reset_index(drop=True, inplace=True)

# feature engineering add a new column price_changes = close - open
combined_df['price_changes'] = combined_df['Close'] - combined_df['Open']

# save data into file
combined_df.to_csv('combined_preprocessed_data.csv', index=False)
print("preprocessing complete. data saved to 'combined_preprocessed_data.csv' ")
