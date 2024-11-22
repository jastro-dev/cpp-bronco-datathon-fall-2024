import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import datetime

for flightType in ['economic', 'premium', 'firstClass']:
    # 1. Load Data
    print(f"Loading data for {flightType}...")
    df = pd.read_csv(f'data/by_type/flights_{flightType}.csv')

    # 2. Data Preparation
    print("Preparing features...")
    # Convert date
    df['date'] = pd.to_datetime(df['date'])

    # Create holiday features
    christmas_2024 = pd.Timestamp('2024-12-25')
    newyear_2024 = pd.Timestamp('2024-12-31')

    df['days_to_christmas'] = abs((df['date'] - christmas_2024).dt.days)
    df['days_to_newyear'] = abs((df['date'] - newyear_2024).dt.days)

    # Get top 10 destinations
    top_destinations = df['to'].value_counts().nlargest(10).index
    df_top = df[df['to'].isin(top_destinations)]

    # 3. Model Training
    print("Training model...")
    # Prepare features
    X = df_top[['days_to_christmas', 'days_to_newyear', 'distance']]
    y = df_top['price']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Validate model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Model R² Score: {r2:.2f}")

    # 4. Generate Predictions
    print("Generating predictions...")
    results = []

    for destination in top_destinations:
        dest_data = df_top[df_top['to'] == destination]
        avg_distance = dest_data['distance'].mean()

        # Predict Christmas price
        christmas_features = np.array([[0, 6, avg_distance]])  # 0 days to Christmas, 6 days to New Year
        christmas_price = model.predict(christmas_features)[0]

        # Predict New Year price
        newyear_features = np.array([[6, 0, avg_distance]])  # 6 days to Christmas, 0 days to New Year
        newyear_price = model.predict(newyear_features)[0]

        # Get best agency
        best_agency = dest_data.groupby('agency')['price'].mean().idxmin()

        results.append({
            'destination': destination,
            'christmas_price': round(christmas_price, 2),
            'newyear_price': round(newyear_price, 2),
            'popularity_rank': dest_data['to'].value_counts().rank(ascending=False)[0],
            'best_agency': best_agency
        })

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('christmas_price')

    # 5. Save Results
    print("Saving results...")
    results_df.to_csv(f'predictions/holiday_predictions_{flightType}.csv', index=False)
    print("\nResults Preview:")
    print(results_df)
