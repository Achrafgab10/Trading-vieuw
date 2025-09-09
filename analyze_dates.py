import pandas as pd
import numpy as np
from datetime import datetime

# Load the data
df = pd.read_csv('data_cleaned.csv', sep=';')
df['Time'] = pd.to_datetime(df['Time'], format='%d.%m.%Y %H:%M', errors='coerce')

print('DATA ANALYSIS REPORT')
print('=' * 50)
print(f'Total rows: {len(df)}')
print(f'Date range: {df["Time"].min()} to {df["Time"].max()}')
print(f'Total days: {(df["Time"].max() - df["Time"].min()).days + 1}')
print()

# Get unique dates
unique_dates = df['Time'].dt.date.unique()
print(f'Unique dates in dataset: {len(unique_dates)}')
print('All dates:')
for i, date in enumerate(sorted(unique_dates)):
    date_count = len(df[df['Time'].dt.date == date])
    print(f'{i+1:2d}. {date} - {date_count} records')

print()
print('Time coverage per day:')
for date in sorted(unique_dates):
    day_data = df[df['Time'].dt.date == date]
    print(f'{date}: {day_data["Time"].min().strftime("%H:%M")} to {day_data["Time"].max().strftime("%H:%M")} ({len(day_data)} records)')

print()
print('Sample of recent data (last 20 rows):')
print(df[['Time', 'Close']].tail(20).to_string())
