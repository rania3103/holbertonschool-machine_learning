#!/usr/bin/env python3
"""preprocess data"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#load data
coinbase_df = pd.read_csv('coinbase.csv')
bitstamp_df = pd.read_csv('bitstamp.csv')
# show first 5 rows
print(coinbase_df.head())
print(bitstamp_df.head())
# delete null values rows
coinbase_df.dropna(inplace=True)
bitstamp_df.dropna(inplace=True)
#check for missing values 
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
coinbase_df = coinbase_df[['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']]
bitstamp_df = bitstamp_df[['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']]
# merge data
combined_df = pd.concat([coinbase_df, bitstamp_df], axis=0, ignore_index=True)
#feature engineering add a new column price_changes = close - open
combined_df['price_changes'] = combined_df['Open'] - combined_df['Close']
# scale data
scaler = MinMaxScaler()
combined_df_scaled = scaler.fit_transform(combined_df)
# convert scaled data to a dataframe
combined_df_scaled = pd.DataFrame(combined_df_scaled, columns=combined_df.columns)
# save data into file
combined_df_scaled.to_csv('combined_scaled_data.csv', index=False)
print("preprocessing complete. sclaed data saved to 'combined_scaled_data.csv' ")

