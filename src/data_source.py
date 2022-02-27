import numpy as np
import pandas as pd

def load_data(macro_target, source_path, rating_filter=None):
    # Load original datasets
    df_stock_characteristics = pd.read_csv(f'{source_path}/Stock_Characteristics_Data_rating.csv')
    df_time_series = pd.read_csv(f'{source_path}/Time_Series_Data.csv')

    # Filter ratings
    if rating_filter is not None:
        df_stock_characteristics = df_stock_characteristics[df_stock_characteristics['rating'].isin(rating_filter)]

    # Use dates with monthly increments
    df_stock_characteristics['datadate'] = pd.to_datetime(df_stock_characteristics['datadate'], format='%Y%m%d')
    df_stock_characteristics['year_month'] = df_stock_characteristics['datadate'].dt.to_period('M')
    df_time_series['yyyymm'] = pd.to_datetime(df_time_series['yyyymm'], format='%Y%m')
    df_time_series['year_month'] = df_time_series['yyyymm'].dt.to_period('M')

    # Forward fill quarterly fundamental updates
    # This step makes sure there is fundamental data for each month
    period_range = pd.period_range(start=df_stock_characteristics['year_month'].min(), end=df_stock_characteristics['year_month'].max())
    df_stock_characteristics = df_stock_characteristics.set_index('year_month').groupby('tic').apply(lambda x: x.reindex(period_range).ffill()).reset_index(0, drop=True)
    df_stock_characteristics[df_stock_characteristics['datadate'].notnull()]
    df_stock_characteristics.index.names = ['year_month']
    df_stock_characteristics.reset_index(level=0, inplace=True)

    # Select the fundamentals that are useful for prediction
    # Calculate their average per month across all companies
    df_fundamental_means = df_stock_characteristics[['year_month', 'ltq', 'atq', 'ceqq', 'cheq', 'chq', 'cshoq', 'dlcq', 'dpq', 'intanq', 'niq', 'pstkq', 'pstkrq', 'seqq', 'txditcq', 'xsgaq', 'capxy', 'epspxy', 'mkvaltq']].groupby('year_month').mean()[2:]

    # Extract the macro target variable for prediction
    df_macro = df_time_series[['year_month', macro_target]]
    df_macro = df_macro.set_index('year_month')

    # Merge predictor and target variables together
    # Shift predictor variables forward 1 month
    # Target variable can now be predicted by last months fundamental variables
    df_data = df_macro.merge(df_fundamental_means.shift(1), left_index=True, right_index=True)
    df_data = df_data.tail(df_data.shape[0] - 1)

    # Split regressors and target in different dataframes
    df_X = df_data[df_fundamental_means.columns]
    df_y = df_data[df_macro.columns]

    return df_X, df_y