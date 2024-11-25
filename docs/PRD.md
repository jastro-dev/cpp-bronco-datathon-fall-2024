# Holiday Flight Price Prediction MVP

## Goal
Predict best cities to travel to for Christmas (Dec 25, 2024) and New Year's (Dec 31, 2024) based on historical flight data.

## Technical Scope
- Simple linear regression model
- Top 10 popular destinations only
- Price and distance as primary features

## Libraries
- pandas
- numpy
- sklearn.linear_model

## Implementation Steps

### 1. Data Preparation (30 mins)
- Load flight data from CSV
- Filter for top 10 destinations
- Create features:
  * Days until holiday
  * Distance
  * Historical prices

### 2. Model Development (30 mins)
- Train linear regression model
- Features:
  * Distance
  * Days until holiday
- Target: Price

### 3. Prediction Generation (30 mins)
- Generate predictions for:
  * Christmas 2024
  * New Year's 2024
- Rank cities by:
  * Predicted price
  * Historical popularity

### 4. Output Format
CSV file with columns:
- Destination
- Predicted Christmas Price
- Predicted New Year Price
- Historical Popularity Rank
- Best Agency

## Success Criteria
- Model explains >60% of price variance
- Predictions for all top 10 destinations
- Clear price rankings for both holidays

## Timeline
Total: 2 hours
- Data prep: 30 mins
- Modeling: 30 mins
- Predictions: 30 mins
- Documentation: 30 mins
