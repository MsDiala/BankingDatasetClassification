# Banking Dataset Classification

This code demonstrates a classification task on a banking dataset. It covers data loading, preprocessing, feature engineering, model training, and result submission. Each step is explained in detail.

```python
# Import necessary libraries
import pandas as pd  # Pandas for data manipulation and analysis
import numpy as np   # NumPy for numerical operations
from sklearn.model_selection import train_test_split, RandomizedSearchCV  # Splitting data and hyperparameter tuning
from sklearn.preprocessing import RobustScaler, OneHotEncoder  # Data preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Ensemble classifiers
from lightgbm import LGBMClassifier  # LightGBM classifier
from sklearn.impute import SimpleImputer  # Handling missing data
from sklearn.compose import ColumnTransformer  # Transforming columns in a dataset
from sklearn.pipeline import Pipeline  # Organizing data processing steps
from sklearn.metrics import accuracy_score  # Measuring model performance
from imblearn.over_sampling import BorderlineSMOTE  # Addressing class imbalance
import os  # Operating system utilities

# Load the training and test data
train_data = pd.read_csv('Train-set.csv')  # Load training data from a CSV file into a table (DataFrame)
test_data = pd.read_csv('Test-set.csv')    # Load test data from a CSV file into a table (DataFrame)

# Separate the 'Target' column from the train data
y_train = train_data['Target']  # Store the outcome we want to predict (labels) in 'y_train'
train_data.drop('Target', axis=1, inplace=True)  # Remove the outcome column from training data

# Combine train and test data for preprocessing
all_data = pd.concat([train_data, test_data], axis=0)  # Combine both datasets for consistent preprocessing

# Feature Engineering: Extract day of the week and create a weekend indicator
try:
    # Convert the 'day' column to a standard date format
    all_data['day'] = pd.to_datetime(all_data['day'])
    # Calculate the day of the week (0-6) from the date and create a new 'day_of_week' feature
    all_data['day_of_week'] = all_data['day'].dt.dayofweek
    # Create a binary feature 'is_weekend' to indicate if it's a weekend (Saturday or Sunday)
    all_data['is_weekend'] = all_data['day_of_week'].isin([5, 6]).astype(int)
    # Remove the original 'day' column as we now have 'day_of_week' and 'is_weekend' features
    all_data.drop('day', axis=1, inplace=True)
except (ValueError, OverflowError, pd._libs.tslibs.np_datetime.OutOfBoundsDatetime):
    # Handle errors due to invalid date formats
    all_data['day'] = pd.to_datetime(all_data['day'], errors='coerce')
    all_data['day_of_week'] = all_data['day'].dt.dayofweek
    all_data['is_weekend'] = all_data['day_of_week'].isin([5, 6]).astype(int)
    all_data.drop('day', axis=1, inplace=True)

# Separate numeric and categorical columns
numeric_cols = all_data.select_dtypes(include=[np.number]).columns  # Identify columns with numerical data
categorical_cols = all_data.select_dtypes(include=[object]).columns  # Identify columns with categorical data

# Create transformers for numeric and categorical columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with the median value of the column
    ('scaler', RobustScaler())  # Scale numeric features to handle outliers
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with the most frequent value of the column
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Convert categorical variables into numerical form
])

# Preprocess the data using the column transformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),  # Apply numeric transformations to numeric columns
    ('cat', categorical_transformer, categorical_cols)  # Apply categorical transformations to categorical columns
])

X_all_preprocessed = preprocessor.fit_transform(all_data)  # Apply preprocessing to the entire dataset

# Handle class imbalance using BorderlineSMOTE
smote = BorderlineSMOTE(sampling_strategy='auto', random_state=42)  # Initialize BorderlineSMOTE for oversampling
X_train_resampled, y_train_resampled = smote.fit_resample(X_all_preprocessed[:train_data.shape[0]], y_train)
# Use BorderlineSMOTE to balance the number of samples for each class

# Create and train optimized models
optimized_rf_model = RandomForestClassifier(n_estimators=150, max_depth=9, random_state=42)
# Initialize a Random Forest classifier with optimized settings
optimized_gb_model = GradientBoostingClassifier(n_estimators=160, learning_rate=0.05, max_depth=7, random_state=42)
# Initialize a Gradient Boosting classifier with optimized settings
optimized_lgbm_model = LGBMClassifier(n_estimators=180, learning_rate=0.1, max_depth=5, random_state=42)
# Initialize a LightGBM classifier with optimized settings

optimized_rf_model.fit(X_train_resampled, y_train_resampled)  # Train the Random Forest model on resampled data
optimized_gb_model.fit(X_train_resampled, y_train_resampled)  # Train the Gradient Boosting model on resampled data
optimized_lgbm_model.fit(X_train_resampled, y_train_resampled)  # Train the LightGBM model on resampled data

# Get predictions using optimized models
test_predictions_rf = optimized_rf_model.predict_proba(X_all_preprocessed[train_data.shape[0]:])[:, 1]
# Generate class probabilities for the positive outcome using the Random Forest model
test_predictions_gb = optimized_gb_model.predict_proba(X_all_preprocessed[train_data.shape[0]:])[:, 1]
# Generate class probabilities for the positive outcome using the Gradient Boosting model
test_predictions_lgbm = optimized_lgbm_model.predict_proba(X_all_preprocessed[train_data.shape[0]:])[:, 1]
# Generate class probabilities for the positive outcome using the LightGBM model

# Combine the predictions using weighted averaging
ensemble_predictions = (0.4 * test_predictions_rf) + (0.4 * test_predictions_gb) + (0.2 * test_predictions_lgbm)
threshold = 0.5  # Set a threshold to determine binary predictions
binary_predictions = (ensemble_predictions >= threshold).astype(int)  # Convert probabilities to binary predictions

# Get the 'id' values from the test_data DataFrame
submission_ids = test_data['id']

# Create binary predictions based on a threshold (e.g., 0.5)
threshold = 0.5
binary_predictions = (ensemble_predictions >= threshold).astype(int)

# Create the submission DataFrame with 'id' and binary 'Target' values
submission_df = pd.DataFrame({'id': submission_ids, 'Target': binary_predictions})

# Save the submission file to CSV
submission_df.to_csv('submission_binary.csv', index=False)
