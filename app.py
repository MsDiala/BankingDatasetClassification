import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

# Load the data
train_data = pd.read_csv('Train-set.csv')  # Replace with your data path

# Handle missing values in the 'balance' column by filling with the median value
median_balance = train_data['balance'].median()
train_data['balance'].fillna(median_balance, inplace=True)

# Extract selected features
selected_features = ['balance', 'month', 'duration', 'pdays', 'poutcome']
X_selected = train_data[selected_features]

# Encode categorical features
encoder = OneHotEncoder(drop='first')
X_encoded = encoder.fit_transform(X_selected[['poutcome']])  # We only need 'poutcome' here, as 'month' is handled separately

# Train a model (you can replace this with your own model training code)
clf = RandomForestClassifier()
clf.fit(X_encoded, train_data['Target'])

# Save the encoder and model as pickle files
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(clf, 'bank_model.pkl')

# Get unique months from the dataset
valid_months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
unique_months = train_data['month'].str.lower().unique()
unique_months = [month for month in unique_months if month in valid_months]

# Streamlit app title and layout
st.set_page_config(page_title='Bank Marketing Prediction', layout='wide')
st.title('Bank Marketing Prediction')

# Apply custom CSS styles
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://www.intelligentcio.com/apac/wp-content/uploads/sites/44/2023/03/AdobeStock_573230856-1-scaled.jpeg');
        background-size: cover;
        background-repeat: no-repeat;
    }
    .stAppContainer {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100vh;
        background-color: rgba(255, 255, 255, 0.9);
    }
    .stForm {
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        background-color: #ffffff;
        width: 400px;
    }
    .stButton {
        background-color: #007bff;
        color: #ffffff;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# User input fields
st.sidebar.header('User Input')

# Add input fields for selected features
user_balance = st.sidebar.number_input("Average Yearly Balance $", value=100000)
user_month = st.sidebar.selectbox("Last Contact Month of Year", unique_months)
user_duration = st.sidebar.number_input("Last Contact Duration (seconds)")
user_pdays = st.sidebar.number_input("Days Since Last Contact")
user_poutcome = st.sidebar.selectbox("Outcome of Previous Campaign", train_data['poutcome'].unique())

# Preprocess user input and make a prediction
if st.sidebar.button("Predict"):
    user_input = pd.DataFrame({
        'balance': [user_balance],
        'month': [user_month],
        'duration': [user_duration],
        'pdays': [user_pdays],
        'poutcome': [user_poutcome]
    })

    user_input_encoded = encoder.transform(user_input[['poutcome']])
    prediction = clf.predict(user_input_encoded)

    # Display the prediction result
    st.sidebar.subheader("Prediction Result")
    if prediction[0] == 1:
        st.sidebar.success("Prediction: Client will subscribe to a term deposit")
    else:
        st.sidebar.error("Prediction: Client will not subscribe to a term deposit")