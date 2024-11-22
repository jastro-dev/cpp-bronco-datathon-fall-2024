import pandas as pd

df = pd.read_csv('data/flights.csv')

df['flightType'].value_counts()

# Split df by flightType and save as separate csv
for flightType in df['flightType'].unique():
    df[df['flightType'] == flightType].to_csv(f'data/flights_{flightType}.csv', index=False)