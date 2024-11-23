# %%
import pandas as pd

# Iterate whole script for data/flights_economic.csv, data/flights_premium.csv, and data/flights_firstClass.csv
df = pd.read_csv(f'data/flights.csv')
df.head()


# %%

import matplotlib.pyplot as plt
import seaborn as sns
# %%

print(df.columns)
# %%

# Filter for December data
df['date'] = pd.to_datetime(df['date'])
december_df = df[df['date'].dt.month == 12].copy()

# Add holiday markers
december_df['holiday'] = 'Regular Day'
december_df.loc[(december_df['date'].dt.day >= 18) & (december_df['date'].dt.day <= 25), 'holiday'] = 'Christmas'
december_df.loc[(december_df['date'].dt.day >= 24) & (december_df['date'].dt.day <= 31), 'holiday'] = 'New Year'

# Holiday Price Analysis
plt.figure(figsize=(15, 8))
sns.boxplot(data=december_df, x=december_df['date'].dt.day, y='price', hue='holiday')
plt.title(f'December Flight Prices by Day')
plt.xlabel('Day of Month')
plt.ylabel('Price ($)')
plt.axvline(x=17.5, color='r', linestyle='--', alpha=0.5)  # Christmas week start
plt.axvline(x=25.5, color='r', linestyle='--', alpha=0.5)  # Christmas week end
plt.axvline(x=23.5, color='g', linestyle='--', alpha=0.5)  # New Year week start
plt.axvline(x=31.5, color='g', linestyle='--', alpha=0.5)  # New Year week end
plt.tight_layout()
plt.savefig(f'visualizations/by_type/holidays_all/december_prices.png')
plt.show()

# Top destinations during holidays
holiday_df = december_df[december_df['holiday'] != 'Regular Day']
plt.figure(figsize=(12, 6))
top_holiday_routes = (
    holiday_df.groupby(['to', 'holiday'])
    .size()
    .unstack(fill_value=0)
    .sort_values('Christmas', ascending=False)
    .head(10)
)

top_holiday_routes.plot(kind='barh', width=0.8)
plt.title(f'Top 10 Destinations During Holidays')
plt.xlabel('Destination')
plt.ylabel('Number of Flights')
plt.tight_layout()
plt.savefig(f'visualizations/by_type/holidays_all/holiday_destinations.png')
plt.show()

# Top destinations during holidays based off price
holiday_df = december_df[december_df['holiday'] != 'Regular Day']
plt.figure(figsize=(12, 6))
top_holiday_routes = (
    holiday_df.groupby(['to', 'holiday'])
    ['price'].mean()
    .unstack(fill_value=0)
    .sort_values('Christmas', ascending=False)
    .head(10)
)
sns.set_palette("GnBu_r")
top_holiday_routes.plot(kind='barh', width=0.8)
plt.title(f'Top 10 Most Expensive Holiday Destinations')
plt.xlabel('Average Price ($)')
plt.ylabel('Destination')
plt.tight_layout()
plt.savefig(f'visualizations/by_type/holidays_all/holiday_destinations_by_price.png')
plt.show()

# Agency pricing during holidays
plt.figure(figsize=(12, 6))
sns.barplot(data=holiday_df, x='agency', y='price', hue='holiday')
plt.title(f'Agency Pricing During Holidays')
plt.xlabel('Agency')
plt.ylabel('Average Price ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'visualizations/by_type/holidays_all/holiday_agency_prices.png')
plt.show()

# Print holiday statistics
print('\nHoliday Price Statistics:')
print(holiday_df.groupby('holiday')['price'].agg(['mean', 'min', 'max', 'count']))
