# %%
import pandas as pd

df = pd.read_csv('data/flights.csv')
df.head()
# %%

def summarize_dataframe(df: pd.DataFrame, df_name: str='DataFrame'):
    """
    Summarize the DataFrame by displaying its shape, missing values,
    data types, and duplicate values.

    Parameters:
        df (pd.DataFrame): The DataFrame to summarize.
        df_name (str): A name for the DataFrame, used in output messages.
    """
    print(f"Summary for {df_name}")
    print(f"Shape: {df.shape}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nDescriptive Statistics:\n{df.describe(include='all')}")
    print(f"\nUnique Values:\n{df.nunique()}")

    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"\nDuplicate Rows: {duplicate_count}")
        print(f"Examples of Duplicate Rows:\n{df[df.duplicated(keep=False)].head()}")
    else:
        print("\nNo Duplicate Rows Found.")

summarize_dataframe(df, 'Travel Dataframe:')

# %%

import matplotlib.pyplot as plt
import seaborn as sns
# %%

print(df.columns)

# %%
# 'travelCode', 'userCode', 'from', 'to', 'flightType', 'price', 'time',       'distance', 'agency', 'date'
categorical_columns = ['flightType', 'agency', 'from', 'to']
numerical_columns = ['price', 'time', 'distance']

# Set global style parameters
plt.style.use('ggplot')
sns.set_palette("Dark2")
sns.set_style("ticks")  # Minimal style with ticks

# Create figure with better spacing
fig = plt.figure(figsize=(16, 13))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# 1. Price vs Distance Scatter Plot
plt.subplot(2, 2, 1)
scatter = sns.scatterplot(data=df, x='distance', y='price',
                            hue='flightType', alpha=0.7,
                            sizes=(20, 200))
plt.title('Flight Prices vs Distance', pad=20, fontsize=12, fontweight='bold')
plt.xlabel('Distance (miles)', fontsize=10)
plt.ylabel('Price ($)', fontsize=10)
sns.despine()
scatter.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 2. Price Distribution by Flight Type
plt.subplot(2, 2, 2)
box = sns.boxplot(data=df, x='flightType', y='price',
                    width=0.6, linewidth=1.5)
plt.title('Price Ranges by Flight Type', pad=20, fontsize=12, fontweight='bold')
plt.xticks(rotation=30)
plt.xlabel('Flight Type', fontsize=10)
plt.ylabel('Price ($)', fontsize=10)
sns.despine()

# 3. Top 10 Popular Routes
plt.subplot(2, 2, 3)
route_counts = (df.groupby(['from', 'to'])
                .size()
                .sort_values(ascending=False)
                .reset_index(name='count')[:10])
route_counts['route'] = route_counts['from'] + ' → ' + route_counts['to']

bars = sns.barplot(data=route_counts, x='count', y='route',
                    palette='viridis')
plt.title('Most Popular Flight Routes', pad=20, fontsize=12, fontweight='bold')
plt.xlabel('Number of Flights', fontsize=10)
plt.ylabel('Route', fontsize=10)
sns.despine()

# Add value labels on bars
for i in bars.containers:
    bars.bar_label(i, padding=5)

# 4. Agency Price Distribution
plt.subplot(2, 2, 4)
violin = sns.violinplot(data=df, x='agency', y='price',
                        inner='box', cut=0)
plt.title('Price Distribution by Agency', pad=20, fontsize=12, fontweight='bold')
plt.xticks(rotation=30)
plt.xlabel('Agency', fontsize=10)
plt.ylabel('Price ($)', fontsize=10)
sns.despine()

# Adjust layout and display
plt.tight_layout()
plt.savefig('visualizations/EDA.png')
plt.show()

# Temporal Analysis (separate figure)
plt.figure(figsize=(12, 6))
temporal = sns.lineplot(data=df, x='date', y='price',
                        ci=None, linewidth=2)
plt.title('Price Trends Over Time', pad=20, fontsize=12, fontweight='bold')
plt.xticks(rotation=30)
plt.xlabel('Date', fontsize=10)
plt.ylabel('Price ($)', fontsize=10)
sns.despine()
plt.tight_layout()
plt.savefig('visualizations/temporal.png')
plt.show()

# Average Price by Agency (separate figure)
plt.figure(figsize=(10, 6))
avg_price = df.groupby('agency')['price'].mean().sort_values(ascending=False)
bars = sns.barplot(x=avg_price.index, y=avg_price.values,
                    palette='viridis')
plt.title('Average Price by Agency', pad=20, fontsize=12, fontweight='bold')
plt.xticks(rotation=30)
plt.xlabel('Agency', fontsize=10)
plt.ylabel('Average Price ($)', fontsize=10)

# Add value labels on bars
for i in bars.containers:
    bars.bar_label(i, fmt='%.0f', padding=5)

sns.despine()
plt.tight_layout()
plt.savefig('visualizations/avg_price.png')
plt.show()
plt.show()

# %%
# Time Series Analysis
from statsmodels.tsa.seasonal import seasonal_decompose

# %%
# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# %%
# Sort by date
df = df.sort_values('date')

# %%
# Create time-based aggregations
daily_avg = df.groupby('date')['price'].agg(['mean', 'count', 'std']).reset_index()
daily_avg.columns = ['date', 'avg_price', 'num_flights', 'price_std']

# %%
# Weekly rolling statistics
weekly_stats = daily_avg.set_index('date').rolling(window='7D').agg({
    'avg_price': 'mean',
    'num_flights': 'sum',
    'price_std': 'mean'
}).reset_index()

# %%
# Monthly aggregation
monthly_stats = df.set_index('date').resample('M').agg({
    'price': ['mean', 'count', 'std'],
    'distance': 'mean'
}).reset_index()

# %%
# Add seasonal features
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['season'] = pd.cut(df['date'].dt.month,
                        bins=[0, 3, 6, 9, 12],
                        labels=['Winter', 'Spring', 'Summer', 'Fall'])

# %%
# Visualization
plt.style.use('ggplot')
fig, axes = plt.subplots(3, 1, figsize=(15, 12))
fig.suptitle('Time Series Analysis of Flight Prices', fontsize=16)

# %%
# Weekly rolling average
axes[0].plot(weekly_stats['date'], weekly_stats['avg_price'], linewidth=2)
axes[0].set_title('7-Day Rolling Average Price')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Average Price ($)')

# %%
# Monthly trends
monthly_stats.plot(x='date', y=('price', 'mean'), ax=axes[1])
axes[1].set_title('Monthly Average Price')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Average Price ($)')

# %%
# Seasonal patterns
sns.boxplot(data=df, x='season', y='price', ax=axes[2])
axes[2].set_title('Price Distribution by Season')
axes[2].set_xlabel('Season')
axes[2].set_ylabel('Price ($)')

plt.tight_layout()
plt.savefig('visualizations/time_series.png')
plt.show()

# %%
# Seasonal Decomposition
# Resample to daily frequency for decomposition
daily_series = df.groupby('date')['price'].mean()
decomposition = seasonal_decompose(daily_series, period=30)

# %%
# Plot decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
decomposition.observed.plot(ax=ax1)
ax1.set_title('Observed')
decomposition.trend.plot(ax=ax2)
ax2.set_title('Trend')
decomposition.seasonal.plot(ax=ax3)
ax3.set_title('Seasonal')
decomposition.resid.plot(ax=ax4)
ax4.set_title('Residual')
plt.tight_layout()
plt.savefig('visualizations/seasonal_decomposition.png')
plt.show()

# %%
# Print summary statistics
print("\nSeasonal Statistics:")
seasonal_stats = df.groupby('season')['price'].agg(['mean', 'std', 'count'])
print(seasonal_stats)

print("\nMonthly Statistics:")
monthly_summary = df.groupby(df['date'].dt.month)['price'].agg(['mean', 'std', 'count'])
print(monthly_summary)

# %%

# Popular Cities Analysis by Season
plt.figure(figsize=(15, 10))

# Get top 10 most frequent destination cities
top_cities = df['to'].value_counts().nlargest(10).index

# Filter data for top cities
top_cities_df = df[df['to'].isin(top_cities)]

# Create seasonal boxplot for top cities, sorted ascending
sns.boxplot(data=top_cities_df, y='to', x='price', hue='season',
            order=top_cities_df.groupby('to')['price'].mean().sort_values(ascending=False).index)
plt.title('Seasonal Price Distribution for Top 10 Destinations')
plt.xlabel('Price ($)')
plt.ylabel('Destination City')
plt.legend(title='Season')
plt.tight_layout()
plt.savefig('visualizations/seasonal_boxplot.png')
plt.show()

# Agency Performance by Season
plt.figure(figsize=(15, 8))
seasonal_agency_avg = df.groupby(['agency', 'season'])['price'].mean().unstack()
seasonal_agency_avg.plot(kind='barh', width=0.8)
plt.title('Average Prices by Agency and Season')
plt.xlabel('Agency')
plt.ylabel('Average Price ($)')
plt.legend(title='Season')
plt.tight_layout()
plt.savefig('visualizations/agency_seasonal_performance.png')
plt.show()

# Print best agencies for each season
print("\nBest Agencies by Season (Lowest Average Prices):")
for season in df['season'].unique():
    season_data = df[df['season'] == season]
    best_agency = season_data.groupby('agency')['price'].mean().idxmin()
    avg_price = season_data.groupby('agency')['price'].mean().min()
    print(f"{season}: {best_agency} (${avg_price:.2f})")

# Best cities to visit by season (based on price and frequency)
print("\nBest Cities to Visit by Season:")
for season in df['season'].unique():
    season_data = df[df['season'] == season]

    # Calculate score based on price (lower is better) and frequency (higher is better)
    city_metrics = season_data.groupby('to').agg({
        'price': ['mean', 'count']
    }).reset_index()

    # Normalize metrics
    price_min = city_metrics[('price', 'mean')].min()
    price_max = city_metrics[('price', 'mean')].max()
    count_min = city_metrics[('price', 'count')].min()
    count_max = city_metrics[('price', 'count')].max()

    city_metrics['score'] = (
        (price_max - city_metrics[('price', 'mean')]) / (price_max - price_min) * 0.7 +
        (city_metrics[('price', 'count')] - count_min) / (count_max - count_min) * 0.3
    )

    best_cities = city_metrics.nlargest(3, 'score')
    print(f"\n{season}:")
    for _, row in best_cities.iterrows():
        print(f"  {row['to']}: Avg Price ${row[('price', 'mean')]:.2f}, "
                f"Flights: {row[('price', 'count')]}")

# %%
