# Customer Intelligence Dashboard Technical Documentation and User Guide


## Table of Contents
- Requirements
- Installation
- Dashboard Overview
- User Guide
- Uploading Data
- Filtering Data
- Data Visualization

### Requirements 

Before using the Customer Intelligence Dashboard, ensure you have the following requirements installed:

Python 3.6+
Required libraries: streamlit, pandas, plotly_express, seaborn, matplotlib
Install the required libraries using the following command:

```bash
pip install streamlit pandas plotly_express seaborn matplotlib
```

### Installation 

Clone the repository containing the Streamlit dashboard code:

```bash
git clone <repository_url>
```

Navigate to the dashboard's directory:

```bash
Copy code
cd <repository_directory>
```

Run the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

### Dashboard Overview 
The Customer Intelligence Dashboard provides users with a powerful tool for exploring and visualizing varied attributes in banking data. Users can upload their own dataset or use a default dataset provided within the repository. The dashboard allows users to filter data based on specific attributes and gain insights through interactive data visualizations.

### User Guide 
#### Uploading Data 
Upon opening the dashboard, you will see a file uploader section labeled ":file_folder: Upload a file." Click the uploader or drag and drop a dataset file (CSV, TXT, XLSX, or XLS) to upload your data. The uploaded file's name will be displayed for reference.

#### Filtering Data <
After uploading a dataset, the dashboard displays two input fields labeled "Minimum Age" and "Maximum Age." Adjust the sliders to set the desired age range for data filtering.

The dashboard provides a sidebar labeled "Choose your filter." This sidebar allows you to filter data based on different attributes such as "Job," "Age," "Marital," "Education," "Default," "Balance," "Previous," and more. Select one or more filter options for each attribute category.

#### Data Visualization 
The dashboard will generate data visualizations based on the selected filters. Interactive histograms and other visualization plots will be displayed for each selected attribute.

The dashboard also presents additional visualizations, such as the relationship between "Balance" and "Job," and the relationship between "Target" and other attributes. These visualizations provide insights into the dataset's characteristics and relationships.


Now, let's proceed to break down the code of the Customer Intelligence Dashboard step by step.

```python
# Import necessary libraries
import streamlit as st 
import pandas as pd
import plotly_express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
```

Import Libraries: 
Import the required libraries, including Streamlit for creating the dashboard's user interface, pandas for data handling, Plotly Express for interactive data visualization, seaborn for additional visualization, and matplotlib for plotting.

```python
# Set Streamlit page configuration
st.set_page_config(page_title="Customer Intelligence Dashboard", page_icon=":chart_with_upwards_trend:", layout="wide")
```

Set Page Configuration: 
Configure the Streamlit page title, icon, and layout style for the dashboard.

```python
# Display title and styling
st.title(":chart_with_upwards_trend: Exploring Varied Attributes in Banking Data")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)
```

Display Title and Styling: 
Display the dashboard title and apply custom styling to enhance the visual appearance.

```python
# Upload dataset or use default
f1 = st.file_uploader(":file_folder: Upload a file", type=(["csv", "txt", "xlsx", "xls"]))
if f1 is not None:
    filename= f1.name
    st.write(filename)
    dataset = pd.read_csv(filename, encoding= "ISO-8859-1")
else:
    os.chdir(r'/Users/macbookpro/BankingDatasetClassification')
    dataset = pd.read_csv("Train-set.csv", encoding= "ISO-8859-1")
```

Upload Dataset: 
Provide users with the option to upload their own dataset or use a default dataset. The uploaded file's name is displayed, and the dataset is loaded using pandas.

```python
# Filter data by age range
col1, col2 = st.columns((2))
StartAGE = dataset['age'].min()
EndAGE = dataset['age'].max()

with col1:
    age1 = st.number_input("Minimum Age", StartAGE)

with col2:
    age2 = st.number_input("Maximum Age", EndAGE)

dataset = dataset[(dataset["age"] >= age1) & (dataset["age"] <= age2)].copy()
```

Filter Data by Age Range: 
Create sliders for users to set the minimum and maximum age values for filtering the dataset.

```python
st.sidebar.header("Choose your filter: ")

with col1:
    # Sidebar for filtering by job
    st.sidebar.header("Filter By:")
    if 'job' in dataset.columns:
        job_options = dataset['job'].unique()
        selected_jobs = st.sidebar.multiselect("Filter by Job:", job_options)
        # Filter the dataset
        filtered_dataset = dataset[dataset['job'].isin(selected_jobs)]
        # Data visualization
        st.write("Data Visualization:")
        fig = px.histogram(filtered_dataset, x='job', title='Job Distribution')
        st.plotly_chart(fig)
    else:
        st.warning("Column 'Job' not found in the dataset.")
    # ... Repeat the same pattern for other filters (age, marital, education, default, balance, previous, poutcome, housing, loan, contact, day, month, duration, campaign, pdays, Target)
```

Filter Data and Data Visualization: 
In a similar manner for each attribute category, create a sidebar for filtering data and generating interactive histograms based on the selected filter options. Warn the user if a specific column is not found in the dataset.

```python
st.title("Relationship between 'Balance' and 'Job'")
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.boxplot(data=dataset, x='job', y='balance')
plt.xticks(rotation=45, ha='right')
plt.xlabel("Job")
plt.ylabel("Balance")
# Display the plot in Streamlit
st.pyplot(plt)
```

Relationship between "Balance" and "Job": 
Display a box plot showing the relationship between job categories and balance values using seaborn and matplotlib. Display the plot using the st.pyplot() function.

```python
# Visualize relationships between Target and other attributes
if 'Target' in dataset.columns:
    target_options = dataset['Target'].unique()
    selected_targets = st.sidebar.multiselect("Select Target Values:", target_options)
    target_filtered_dataset = dataset[dataset['Target'].isin(selected_targets)]
    # ... Repeat the same pattern for other visualizations (relationship between Target and Job, Target and Contact, Target and Marital)
else:
    st.warning("Column 'Target' not found in the dataset.")
```

Visualizing Relationships with "Target": 
Allow users to select specific target values for filtering. Generate visualizations to explore relationships between "Target" and other attributes, such as "Job," "Contact," and "Marital."
