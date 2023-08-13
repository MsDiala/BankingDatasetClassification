# Banking Dataset Classification Technical Documentation

Introduction
Welcome to the comprehensive technical documentation for the Banking Dataset Classification project. This guide will provide a detailed step-by-step explanation of the code, along with descriptive explanations for each key component. The project's objective is to create a robust ensemble model that predicts specific financial decisions made by banking customers. The code encompasses essential stages, including data loading, preprocessing, feature engineering, model training, prediction, and submission file creation.

## Requirements
Before you begin, make sure to install the required libraries using the following command:

``` bash
pip install pandas numpy scikit-learn lightgbm imbalanced-learn statsmodels
```

These libraries are essential for tasks such as data manipulation, numerical operations, machine learning, handling class imbalance, and statistical modeling.

## Step-by-Step Explanation
 
### 1. Importing Libraries
In this step, we import the necessary libraries that will be used throughout the code. Each library serves a specific purpose, from data manipulation to machine learning model training.

```python
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical operations
from sklearn.model_selection import train_test_split, RandomizedSearchCV  # Data splitting and hyperparameter tuning
from sklearn.preprocessing import RobustScaler, OneHotEncoder  # Data scaling and one-hot encoding
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Ensemble classifiers
from lightgbm import LGBMClassifier  # Gradient boosting framework
from sklearn.impute import SimpleImputer  # Handling missing data
from sklearn.compose import ColumnTransformer  # Column-wise transformations
from sklearn.pipeline import Pipeline  # Data preprocessing pipeline
from sklearn.metrics import accuracy_score  # Model evaluation
from imblearn.over_sampling import BorderlineSMOTE  # Handling class imbalance
import os  # Operating system utilities
import statsmodels.api as sm  # Statistical modeling
```

### 2. Loading Data
In this step, we load the training and test datasets from CSV files using the pandas library.

```python
train_data = pd.read_csv('Train-set.csv')
test_data = pd.read_csv('Test-set.csv')
```

### 3. Target Separation
Here, we separate the target variable ('Target') from the training data to use it later for training the models.

```python
y_train = train_data['Target']
train_data.drop('Target', axis=1, inplace=True)
```

### 4. Feature Engineering
Feature engineering involves creating new features from existing data. In this case, we extract the day of the week and create a weekend indicator based on the 'day' column.

```python
# Convert 'day' column to datetime
all_data['day'] = pd.to_datetime(all_data['day'])

# Extract day of the week and create 'day_of_week' feature
all_data['day_of_week'] = all_data['day'].dt.dayofweek

# Create binary indicator for the weekend (Saturday and Sunday)
all_data['is_weekend'] = all_data['day_of_week'].isin([5, 6]).astype(int)

# Drop the original 'day' column
all_data.drop('day', axis=1, inplace=True)
```

### 5. Column Separation

We categorize columns into numeric and categorical types to facilitate preprocessing.

```python
numeric_cols = all_data.select_dtypes(include=[np.number]).columns
categorical_cols = all_data.select_dtypes(include=[object]).columns
```

### 6. Data Preprocessing Pipelines

Here, we create two preprocessing pipelines for numeric and categorical columns.

``` python
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
```

### 7. Column Transformation

We use the ColumnTransformer to apply the preprocessing pipelines to numeric and categorical columns.

```python
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

X_all_preprocessed = preprocessor.fit_transform(all_data)
```

### 8. Handling Class Imbalance

Class imbalance is a common issue in machine learning. Here, we use the BorderlineSMOTE technique to balance class distribution.

```python
smote = BorderlineSMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_all_preprocessed[:train_data.shape[0]], y_train)
```

### 9. Model Initialization and Training

We initialize and train three optimized classification models: RandomForestClassifier, GradientBoostingClassifier, and LGBMClassifier.

```python
optimized_rf_model = RandomForestClassifier(n_estimators=150, max_depth=9, random_state=42)
optimized_gb_model = GradientBoostingClassifier(n_estimators=160, learning_rate=0.05, max_depth=7, random_state=42)
optimized_lgbm_model = LGBMClassifier(n_estimators=180, learning_rate=0.1, max_depth=5, random_state=42)

optimized_rf_model.fit(X_train_resampled, y_train_resampled)
optimized_gb_model.fit(X_train_resampled, y_train_resampled)
optimized_lgbm_model.fit(X_train_resampled, y_train_resampled)
```

### 10. Predictions and Ensemble

We generate predictions using the trained models and create an ensemble prediction through weighted averaging.

``` python
test_predictions_rf = optimized_rf_model.predict_proba(X_all_preprocessed[train_data.shape[0]:])[:, 1]
test_predictions_gb = optimized_gb_model.predict_proba(X_all_preprocessed[train_data.shape[0]:])[:, 1]
test_predictions_lgbm = optimized_lgbm_model.predict_proba(X_all_preprocessed[train_data.shape[0]:])[:, 1]

ensemble_predictions = (0.4 * test_predictions_rf) + (0.4 * test_predictions_gb) + (0.2 * test_predictions_lgbm)
threshold = 0.5
binary_predictions = (ensemble_predictions >= threshold).astype(int)
```

### 11. Backward Elimination

For those interested in feature selection using backward elimination, we can integrate the code in the right place:

```python
# Backward Elimination
def backward_elimination(X, y, threshold=0.05):
    num_features = X.shape[1]
    for i in range(num_features):
        ols_model = sm.OLS(y, X).fit()
        max_p_value = max(ols_model.pvalues)
        if max_p_value > threshold:
            max_p_idx = np.argmax(ols_model.pvalues)
            X = np.delete(X, max_p_idx, axis=1)
        else:
            break
    return X

# Apply Backward Elimination to select features
X_selected = backward_elimination(X_train_resampled, y_train_resampled)
```

### 12. Submission File Creation

We create the final submission DataFrame by combining 'id' values from the test data with the binary predictions.

```python
submission_ids = test_data['id']
submission_df = pd.DataFrame({'id': submission_ids, 'Target': binary_predictions})
```

### 13. Save Submission File

Finally, we save the submission DataFrame as a CSV file.

```python
submission_df.to_csv('submission_binary.csv', index=False)
```




