# Bank Marketing Prediction App Technical Documentation and User Guide


## Table of Contents
- Requirements
- Installation
- App Overview
- User Guide
- Input Fields
- Making Predictions

### Requirements

Before using the Bank Marketing Prediction App, ensure you have the following requirements installed:

Python 3.6+
Required libraries: streamlit, numpy, pandas, joblib, scikit-learn
Install the required libraries using the following command:

```bash
pip install streamlit numpy pandas joblib scikit-learn
```

### Installation

Clone the repository containing the Streamlit app code:

```bash
git clone <repository_url>
```

Navigate to the app's directory:

```bash
cd <repository_directory>
```

Run the Streamlit app:

```bash
streamlit run app.py
```

### App Overview 

The Bank Marketing Prediction App is designed to provide users with a convenient way to predict whether a client will subscribe to a term deposit based on specific input features. The app utilizes machine learning techniques and a trained model to make predictions. Users can interact with the app through an intuitive user interface, allowing them to input various features and obtain instant predictions.

### User Guide

#### Input Fields 

The user input fields are located in the sidebar on the left-hand side of the app. Each input field corresponds to a feature used for prediction. Here's a breakdown of the input fields:

- Average Yearly Balance: Enter the average yearly balance of the client in dollars.
Last Contact Month of Year: Select the last contact month of the year from the dropdown menu.
- Last Contact Duration (seconds): Enter the duration of the last contact with the client in seconds.
- Days Since Last Contact: Enter the number of days since the last contact with the client.
- Outcome of Previous Campaign: Select the outcome of the previous marketing campaign from the dropdown menu.

#### Making Predictions 
Fill in the input fields with the desired values.
Click the "Predict" button located at the bottom of the sidebar.
The app will process your input, make a prediction using the trained model, and display the result in the sidebar.
If the prediction is positive, the app will display a success message indicating that the client is likely to subscribe to a term deposit. If the prediction is negative, the app will display an error message indicating that the client is unlikely to subscribe to a term deposit.


Now, let's proceed to break down the code of the Bank Marketing Prediction App step by step.

```python
# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
```

Import the required libraries: 
We start by importing the necessary libraries, including Streamlit for creating the app's user interface, numpy and pandas for data handling, joblib for saving and loading models, OneHotEncoder for feature encoding, and RandomForestClassifier for model prediction.

```python
# Load the data
train_data = pd.read_csv('Train-set.csv')  # Replace with your data path
```

Load the data: 
Load the training data from a CSV file using pandas. Replace 'Train-set.csv' with the appropriate file path.

```python
# Handle missing values in the 'balance' column by filling with the median value
median_balance = train_data['balance'].median()
train_data['balance'].fillna(median_balance, inplace=True)
```

Handle missing values: 
Fill missing values in the 'balance' column with the median value.

```python
# Extract selected features
selected_features = ['balance', 'month', 'duration', 'pdays', 'poutcome']
X_selected = train_data[selected_features]
```

Extract features: 
Create a subset of the data containing only the selected features.

```python
# Encode categorical features
encoder = OneHotEncoder(drop='first')
X_encoded = encoder.fit_transform(X_selected[['poutcome']])
```

Encode categorical features:
 Use OneHotEncoder to encode the categorical feature 'poutcome'. We use the 'drop' parameter to drop the first category to avoid multicollinearity.

```python
# Train a model (you can replace this with your own model training code)
clf = RandomForestClassifier()
clf.fit(X_encoded, train_data['Target'])
```

Train a model: 
Train a RandomForestClassifier using the encoded features and the target variable 'Target'.

```python
# Save the encoder and model as pickle files
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(clf, 'bank_model.pkl')
```

Save the model: 
Save the encoder and trained model as pickle files for later use.

```python
# Get unique months from the dataset
valid_months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
unique_months = train_data['month'].str.lower().unique()
unique_months = [month for month in unique_months if month in valid_months]
```

Get unique months: 
Create a list of valid months based on the dataset's unique 'month' values.

```python
# Streamlit app title and layout
st.set_page_config(page_title='Bank Marketing Prediction', layout='wide')
st.title('Bank Marketing Prediction')
```

Set app title and layout: 
Configure the Streamlit app title and layout. The set_page_config function sets the page title and layout style to 'wide', and title sets the main app title.

```python
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
```

Apply custom CSS styles: 
Use the .markdown() function to apply custom CSS styles to different elements of the app's user interface. The styles include background images, container styling, form styling, and button styling for an enhanced visual experience.

```python
# User input fields
st.sidebar.header('User Input')
```

User input fields: 
Display a header in the sidebar indicating the user input fields section.

```python
# Add input fields for selected features
user_balance = st.sidebar.number_input("Average Yearly Balance $", value=100000)
user_month = st.sidebar.selectbox("Last Contact Month of Year", unique_months)
user_duration = st.sidebar.number_input("Last Contact Duration (seconds)")
user_pdays = st.sidebar.number_input("Days Since Last Contact")
user_poutcome = st.sidebar.selectbox("Outcome of Previous Campaign", train_data['poutcome'].unique())
```

Input fields: 
Create input fields in the sidebar for users to provide input values for the selected features. Users can enter values for the average yearly balance, last contact month, contact duration, days since last contact, and outcome of the previous campaign.

```python
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
```

Preprocess user input and prediction: 
When the "Predict" button is clicked, the app preprocesses the user's input, encodes the categorical feature, and makes a prediction using the trained classifier.

```python
# Display the prediction result
st.sidebar.subheader("Prediction Result")
if prediction[0] == 1:
    st.sidebar.success("Prediction: Client will subscribe to a term deposit")
else:
    st.sidebar.error("Prediction: Client will not subscribe to a term deposit")
```

Display prediction result: 
Display the prediction result in the sidebar. If the prediction is positive (1), display a success message indicating that the client will subscribe to a term deposit. If the prediction is negative (0), display an error message indicating that the client will not subscribe to a term deposit.


