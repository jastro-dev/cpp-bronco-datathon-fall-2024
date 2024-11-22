import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import datetime


def clean_and_validate_data(df, flight_type):
    # Remove outliers based on price
    q1 = df['price'].quantile(0.25)
    q3 = df['price'].quantile(0.75)
    iqr = q3 - q1
    price_min = q1 - 1.5 * iqr
    price_max = q3 + 1.5 * iqr

    # Set minimum prices based on flight type
    min_prices = {'economic': 400, 'premium': 600, 'firstClass': 800}

    df = df[(df['price'] >= max(price_min, min_prices[flight_type])) & (df['price'] <= price_max)]

    return df


def enforce_price_constraints(price, flight_type):
    # Enforce minimum and maximum prices
    price_constraints = {'economic': (500, 1000), 'premium': (700, 1400), 'firstClass': (900, 2000)}

    min_price, max_price = price_constraints[flight_type]
    return np.clip(price, min_price, max_price)


for flightType in ['economic', 'premium', 'firstClass']:
    print(f'\nProcessing {flightType} flights...')

    # 1. Load and Clean Data
    df = pd.read_csv(f'data/by_type/flights_{flightType}.csv')
    df = clean_and_validate_data(df, flightType)

    # 2. Feature Engineering
    df['date'] = pd.to_datetime(df['date'])
    christmas_2024 = pd.Timestamp('2024-12-25')
    newyear_2024 = pd.Timestamp('2024-12-31')

    df['days_to_christmas'] = abs((df['date'] - christmas_2024).dt.days)
    df['days_to_newyear'] = abs((df['date'] - newyear_2024).dt.days)

    # Get top 10 destinations
    top_destinations = df['to'].value_counts().nlargest(10).index
    df_top = df[df['to'].isin(top_destinations)]

    # 3. Feature Scaling and Model Training
    X = df_top[['days_to_christmas', 'days_to_newyear', 'distance']]
    y = df_top['price']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Validate model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f'Model R² Score: {r2:.2f}')

    # 4. Generate Predictions
    results = []
    for destination in top_destinations:
        dest_data = df_top[df_top['to'] == destination]
        avg_distance = dest_data['distance'].mean()

        # Prepare features for prediction
        christmas_features = np.array([[0, 6, avg_distance]])
        newyear_features = np.array([[6, 0, avg_distance]])

        # Scale features
        christmas_features_scaled = scaler.transform(christmas_features)
        newyear_features_scaled = scaler.transform(newyear_features)

        # Generate and constrain predictions
        christmas_price = model.predict(christmas_features_scaled)[0]
        newyear_price = model.predict(newyear_features_scaled)[0]

        christmas_price = enforce_price_constraints(christmas_price, flightType)
        newyear_price = enforce_price_constraints(newyear_price, flightType)

        # Get best agency
        best_agency = dest_data.groupby('agency')['price'].mean().idxmin()

        results.append(
            {
                'destination': destination,
                'christmas_price': round(christmas_price, 2),
                'newyear_price': round(newyear_price, 2),
                'popularity_rank': dest_data['to'].value_counts().rank(ascending=False)[0],
                'best_agency': best_agency,
            }
        )

    # 5. Save Results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('christmas_price')
    results_df.to_csv(f'predictions/holiday_predictions_{flightType}.csv', index=False)
    print('\nResults Preview:')
    print(results_df)
