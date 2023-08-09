import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif

# Load the data
train_data = pd.read_csv('Train-set.csv')  # Replace with your data path

# Extract categorical columns
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month', 'poutcome']
X_categorical = train_data[categorical_columns]

# Encode categorical features
encoder = OneHotEncoder(drop='first')
X_encoded = encoder.fit_transform(X_categorical)

# Select important features
selector = SelectKBest(f_classif, k=10)
selected_features = selector.fit(X_encoded, train_data['Target']).get_support(indices=True)
X_encoded_selected = X_encoded[:, selected_features]

# Train a model (you can replace this with your own model training code)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_encoded_selected, train_data['Target'])

# Save the encoder and model as pickle files
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(clf, 'bank_model.pkl')

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
    .st-br {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# User input fields
st.sidebar.header('User Input')

# Add input fields for user's age, job, marital status, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome

user_job = st.sidebar.selectbox("Type of job", train_data['job'].unique())
user_marital = st.sidebar.selectbox("Marital status", train_data['marital'].unique())
user_education = st.sidebar.selectbox("Education level", train_data['education'].unique())
user_default = st.sidebar.selectbox("Has credit in default?", train_data['default'].unique())
user_balance = st.sidebar.number_input("Average yearly balance in euros")
user_housing = st.sidebar.selectbox("Has housing loan?", train_data['housing'].unique())
user_loan = st.sidebar.selectbox("Has personal loan?", train_data['loan'].unique())
user_contact = st.sidebar.selectbox("Contact communication type", train_data['contact'].unique())
user_day = st.sidebar.selectbox("Last contact day of the week", train_data['day'].unique())
user_month = st.sidebar.selectbox("Last contact month of year", train_data['month'].unique())
user_duration = st.sidebar.number_input("Last contact duration, in seconds")
user_campaign = st.sidebar.number_input("Number of contacts performed during this campaign")
user_pdays = st.sidebar.number_input("Number of days passed by after the client was last contacted")
user_previous = st.sidebar.number_input("Number of contacts performed before this campaign")
user_poutcome = st.sidebar.selectbox("Outcome of the previous marketing campaign", train_data['poutcome'].unique())

# Preprocess user input and make a prediction
if st.sidebar.button("Predict"):
    user_categorical_input = np.array([user_job, user_marital, user_education, user_default,
                                       user_housing, user_loan, user_contact, user_day,
                                       user_month, user_poutcome]).reshape(1, -1)
    user_input_encoded = encoder.transform(user_categorical_input)
    user_input_selected = user_input_encoded[:, selected_features]
    prediction = clf.predict(user_input_selected)

    # Display the prediction result
    st.sidebar.subheader("Prediction Result")
    if prediction[0] == 1:
        st.sidebar.write("Prediction: Client will subscribe to a term deposit")
    else:
        st.sidebar.write("Prediction: Client will not subscribe to a term deposit")
