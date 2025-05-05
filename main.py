import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import optuna
from scipy.optimize import minimize

# Set Streamlit configurations
st.set_page_config(page_title="Dynamic Pricing Suite", layout="wide")

# Orange Divider Function
def orange_divider():
    st.markdown("<hr style='border:2px solid orange'>", unsafe_allow_html=True)

# Streamlit UI for dataset upload
st.title("Dynamic Pricing Suite")
st.write("Upload your flight data as a CSV or Excel file to predict and optimize prices.")

uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Load dataset
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.write("### Dataset Overview")
    st.dataframe(df.head())
    orange_divider()

    # Function to clean and prepare flight data
    def clean_flight_data(df):
        df['Airline-Name'] = df['Airline-Class'].str.split('\n').str[0].str.strip()
        df['Class'] = df['Airline-Class'].str.split('\n').str[-1].str.strip()
        df['Date of Journey'] = pd.to_datetime(df['Date of Journey'], format='%d/%m/%Y')
        df['Date of Booking'] = pd.to_datetime(df['Date of Booking'], format='%d/%m/%Y')
        df['days_before_flight'] = (df['Date of Journey'] - df['Date of Booking']).dt.days
        df['journey_day'] = df['Date of Journey'].dt.day
        df['journey_day_name'] = df['Date of Journey'].dt.day_name()
        df['Departure City'] = df['Departure Time'].str.split('\n').str[1].str.strip()
        df['Arrival City'] = df['Arrival Time'].str.split('\n').str[1].str.strip()
        df['Total Stops'] = df['Total Stops'].str.replace(r'\n\s*\t*', '', regex=True).str.replace(r'(stop).*', r'\1', regex=True)
        df['Departure_Time'] = df['Departure Time'].str.split('\n').str[0].str.strip()
        df['Arrival_Time'] = df['Arrival Time'].str.split('\n').str[0].str.strip()
        df['Duration'] = df['Duration'].str.extract(r'(\d+)h (\d+)m').astype(float).apply(lambda x: round(x[0] + x[1] / 60, 4), axis=1)
        df['arrival_time'] = pd.to_datetime(df['Arrival_Time'], format='%H:%M').dt.hour
        df['arrival_category'] = df['arrival_time'].apply(lambda x: 'Before 7pm' if x < 19 else 'After 7pm')
        df.drop(['Date of Booking', 'Date of Journey', 'Airline-Class', 'Departure Time', 'Arrival Time', 'arrival_time'], axis=1, inplace=True)
        df['Price'] = df['Price'].replace(',', '', regex=True).astype(int)
        return df

    # Cleaning the data
    df = clean_flight_data(df)
    st.write("### Cleaned Data Preview")
    st.dataframe(df.head())
    orange_divider()

    # Feature Engineering
    def feat_eng(df):
        df['Route'] = df['Departure City'] + ' -> ' + df['Arrival City']
        df.drop(['Departure City', 'Arrival City'], axis=1, inplace=True)
        df['Departure_Time_hr'] = pd.to_datetime(df['Departure_Time'], format='%H:%M').dt.hour
        df['Arrival_Time_hr'] = pd.to_datetime(df['Arrival_Time'], format='%H:%M').dt.hour
        df.drop(['Departure_Time', 'Arrival_Time'], axis=1, inplace=True)
        return df

    df = feat_eng(df)
    
    # Filter Data for Air India, Economy Class, and specific route
    df = df[(df['Airline-Name'] == 'Air India') & (df['Class'] == 'ECONOMY') & (df['Route'] == 'Delhi -> Mumbai')]
    df.drop(['Class', 'Airline-Name', 'Route'], inplace=True, axis=1)

    # Encoding Categorical Data
    def enc_data(df):
        df = pd.get_dummies(df, columns=['Total Stops'], prefix='Stops', dtype=int)
        
        # Define the frequency map globally so it can be reused for custom inputs
        global frequency_map
        frequency_map = df['journey_day_name'].value_counts(normalize=True).to_dict()
        
        df['journey_day_name_FreqEnc'] = df['journey_day_name'].map(frequency_map)
        df.drop(['journey_day_name'], axis=1, inplace=True)
        label_encoder = LabelEncoder()
        df['arrival_category'] = label_encoder.fit_transform(df['arrival_category'])
        return df

    enc_df = enc_data(df)

    # Predictor and Target Variables
    X = enc_df.drop(['Price'], axis=1)
    y = enc_df['Price']
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling the Features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # XGBoost Model Training
    def xgboostmodel(X_train, y_train):
        model = xgb.XGBRegressor(learning_rate=0.063, max_depth=7, subsample=0.775, colsample_bytree=0.768, min_child_weight=1)
        model.fit(X_train, y_train)
        return model

    model = xgboostmodel(X_train, y_train)

    # Predictions and Evaluation
    predicted_prices = model.predict(X_test)
    r2 = r2_score(y_test, predicted_prices)
    rmse = mean_squared_error(y_test, predicted_prices, squared=False)

    st.write(f"### Model Evaluation")
    st.write(f"R² Score: {r2}")
    st.write(f"RMSE: {rmse}")
    orange_divider()

    # Feature Importance
    feature_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    st.write("### Feature Importance")
    st.dataframe(feature_importance_df)
    orange_divider()

    # Revenue Optimization
    revenue_before = np.sum(predicted_prices)

    def objective_function(prices):
        revenue = -np.sum(prices)
        return revenue

    def constraint(prices, predicted_prices, min_revenue):
        return np.concatenate([prices - predicted_prices, predicted_prices * 1.2 - prices, [np.sum(prices) - min_revenue]])

    initial_prices = predicted_prices
    min_revenue = np.sum(predicted_prices) * 0.9
    bounds = [(price, price * 1.2) for price in predicted_prices]
    result = minimize(objective_function, initial_prices, constraints={'type': 'ineq', 'fun': constraint, 'args': (predicted_prices, min_revenue)}, bounds=bounds)
    optimized_prices = result.x
    revenue_after = np.sum(optimized_prices)

    st.write("### Revenue Optimization")
    st.write(f"Revenue before optimization: {revenue_before}")
    st.write(f"Revenue after optimization: {revenue_after}")
    orange_divider()

    # **Price Comparison Table with Merged Data**
    st.write("### Price Comparison: Predicted vs Optimized Prices with Features")
    
    # Merge input features with predicted and optimized prices
    X_test_df = pd.DataFrame(X_test, columns=X.columns)  # Convert scaled X_test back to DataFrame
    X_test_df['Predicted Price'] = predicted_prices
    X_test_df['Optimized Price'] = optimized_prices
    
    st.dataframe(X_test_df)
    
    orange_divider()

    # KDE Plot
    st.write("### Price Distribution Plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(x=optimized_prices, label='Optimized Prices', color='green', ax=ax)
    sns.kdeplot(x=predicted_prices, label='Predicted Prices', color='red', ax=ax)
    plt.xlabel('Price')
    plt.ylabel('Density')
    plt.title('Distribution of Predicted and Optimized Prices')
    st.pyplot(fig)
    orange_divider()

    ### New Feature: User Inputs for Custom Prediction
    st.write("## Predict Price Based on Custom Features")

    # User inputs for custom prediction
    stops = st.selectbox("Select Total Stops", options=['Non-stop', '1 stop', '2+ stops'])
    days_before_flight = st.slider("Days Before Flight", min_value=0, max_value=365, value=30)
    journey_day = st.slider("Journey Day (1-31)", min_value=1, max_value=31, value=15)
    journey_day_name = st.selectbox("Day of the Week", options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    departure_time_hr = st.slider("Departure Hour (24-hour format)", min_value=0, max_value=23, value=10)
    arrival_time_hr = st.slider("Arrival Hour (24-hour format)", min_value=0, max_value=23, value=12)
    arrival_category = st.selectbox("Arrival Category", options=['Before 7pm', 'After 7pm'])

    # Get custom Duration input (assumed for simplicity, this could be derived from more detailed user inputs)
    duration_hr = st.slider("Flight Duration Hours", min_value=1, max_value=24, value=2)
    duration_min = st.slider("Flight Duration Minutes", min_value=0, max_value=59, value=30)
    duration = round(duration_hr + (duration_min / 60), 4)

    # Dynamically match the columns from the training data
    input_data = pd.DataFrame(columns=X.columns)  # Create dataframe with same columns as training data

    # Set values for the features, making sure all columns are present
    input_data.loc[0, 'days_before_flight'] = days_before_flight
    input_data.loc[0, 'journey_day'] = journey_day
    input_data.loc[0, 'journey_day_name_FreqEnc'] = frequency_map.get(journey_day_name, 0)
    input_data.loc[0, 'Departure_Time_hr'] = departure_time_hr
    input_data.loc[0, 'Arrival_Time_hr'] = arrival_time_hr
    input_data.loc[0, 'arrival_category'] = 0 if arrival_category == 'Before 7pm' else 1
    input_data.loc[0, 'Duration'] = duration

    # Dynamically match the stops encoding based on the columns available in X
    stops_encoded = {col: 0 for col in X.columns if 'Stops_' in col}  # Initialize all Stops columns to 0
    if stops == 'Non-stop' and 'Stops_non-stop' in stops_encoded:
        stops_encoded['Stops_non-stop'] = 1
    elif stops == '1 stop' and 'Stops_1-stop' in stops_encoded:
        stops_encoded['Stops_1-stop'] = 1
    elif stops == '2+ stops' and 'Stops_2-stop' in stops_encoded:
        stops_encoded['Stops_2-stop'] = 1

    for stop_col, val in stops_encoded.items():
        input_data.loc[0, stop_col] = val

    # Fill any missing columns with 0 (if training had additional features)
    input_data = input_data.fillna(0)

    # Scaling the input data
    input_data_scaled = scaler.transform(input_data)

    # Predicting the price
    custom_predicted_price = model.predict(input_data_scaled)
    custom_optimized_price = custom_predicted_price * 1.1  # For simplicity, increasing by 10% for optimization

    st.write("### Predicted and Optimized Prices with Corresponding Features")

    # Merge the input data with the predicted and optimized prices for clear output
    merged_output = input_data.copy()
    merged_output['Predicted Price'] = custom_predicted_price
    merged_output['Optimized Price'] = custom_optimized_price

    # Display the merged dataframe with predicted and optimized prices
    st.dataframe(merged_output)
    orange_divider()

    st.write("The above table shows the predicted and optimized prices alongside the features you selected for your flight scenario.")
