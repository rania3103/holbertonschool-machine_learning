#!/usr/bin/env python3
"""a function  that takes two pd.DataFrame objects and
Indexes both dataframes on their Timestamp columns,
Includes all timestamps from df2 (bitstamp) up to and
including timestamp 1417411920,
Concatenates the selected rows from df2 to the top of df1 (coinbase) and
Adds keys to the concatenated data, labeling the rows
from df2 as bitstamp and the rows from df1 as coinbase"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """Returns the concatenated pd.DataFrame"""
    df1, df2 = index(df1), index(df2)
    df2 = df2[df2.index <= 1417411920]
    return pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
