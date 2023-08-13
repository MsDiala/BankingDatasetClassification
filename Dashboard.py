import streamlit as st 
import pandas as pd
import plotly_express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Customer Intelligence Dashboard", page_icon=":chart_with_upwards_trend:", layout="wide")

st.title(":chart_with_upwards_trend: Exploring Varied Attributes in Banking Data")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

f1 = st.file_uploader(":file_folder: Upload a file", type=(["csv", "txt", "xlsx", "xls"]))
if f1 is not None:
    filename= f1.name
    st.write(filename)
    dataset = pd.read_csv(filename, encoding= "ISO-8859-1")
else:
    os.chdir(r'/Users/macbookpro/BankingDatasetClassification')
    dataset = pd.read_csv("Train-set.csv", encoding= "ISO-8859-1")

col1, col2 = st.columns((2))

StartAGE = dataset['age'].min()
EndAGE = dataset['age'].max()

with col1:
    age1 = st.number_input("Minimum Age", StartAGE)

with col2:
    age2 = st.number_input("Maximum Age", EndAGE)

dataset = dataset[(dataset["age"] >= age1) & (dataset["age"] <= age2)].copy()

st.sidebar.header("Choose your filter: ")

with col1:
    # Sidebar for filtering by job
    st.sidebar.header("Filter By:")
    if 'job' in dataset.columns:
        job_options = dataset['job'].unique()
        selected_jobs = st.sidebar.multiselect("Filter by Job:", job_options)
        
        # Filter the dataset
        filtered_dataset = dataset[dataset['job'].isin(selected_jobs)]

        # Display the filtered data
        #st.write("Filtered Data:")
        #st.dataframe(filtered_dataset)

        # Data visualization
        st.write("Data Visualization:")
        fig = px.histogram(filtered_dataset, x='job', title='Job Distribution')
        st.plotly_chart(fig)
    else:
        st.warning("Column 'Job' not found in the dataset.")


    # Sidebar for filtering by age

    if 'age' in dataset.columns:
        age_options = dataset['age'].unique()
        selected_ages = st.sidebar.multiselect("Filter by Age:", age_options)
        
        
    
        # Data visualization
        st.write("Data Visualization:")
        fig = px.histogram(filtered_dataset, x='age', title='Age Distribution')
        st.plotly_chart(fig)
    else:
        st.warning("Column 'Age' not found in the dataset.")


    # Sidebar for filtering by age

    if 'marital' in dataset.columns:
        age_options = dataset['marital'].unique()
        selected_ages = st.sidebar.multiselect("Filter by Marital:", age_options)
        
    

        # Data visualization
        st.write("Data Visualization:")
        fig = px.histogram(filtered_dataset, x='marital', title='Marital Distribution')
        st.plotly_chart(fig)
    else:
        st.warning("Column 'Marital' not found in the dataset.")

    # Sidebar for filtering by age

    if 'education' in dataset.columns:
        age_options = dataset['education'].unique()
        selected_ages = st.sidebar.multiselect("Filter by Education:", age_options)
        


        # Data visualization
        st.write("Data Visualization:")
        fig = px.histogram(filtered_dataset, x='education', title='Education Distribution')
        st.plotly_chart(fig)
    else:
        st.warning("Column 'Education' not found in the dataset.")




    if 'default' in dataset.columns:
        age_options = dataset['default'].unique()
        selected_ages = st.sidebar.multiselect("Filter by Default:", age_options)

        # Data visualization
        st.write("Data Visualization:")
        fig = px.histogram(filtered_dataset, x='default', title='Default Distribution')
        st.plotly_chart(fig)
    else:
        st.warning("Column 'Default' not found in the dataset.")



    if 'balance' in dataset.columns:
        age_options = dataset['balance'].unique()
        selected_ages = st.sidebar.multiselect("Filter by balance:", age_options)
        
        # Data visualization
        st.write("Data Visualization:")
        fig = px.histogram(filtered_dataset, x='balance', title='balance Distribution')
        st.plotly_chart(fig)
    else:
        st.warning("Column 'balance' not found in the dataset.")

    
    if 'previous' in dataset.columns:
        age_options = dataset['previous'].unique()
        selected_ages = st.sidebar.multiselect("Filter by previous:", age_options)

        # Data visualization
        st.write("Data Visualization:")
        fig = px.histogram(filtered_dataset, x='previous', title='previous Distribution')
        st.plotly_chart(fig)
    else:
        st.warning("Column 'previous' not found in the dataset.")


    if 'poutcome' in dataset.columns:
        age_options = dataset['poutcome'].unique()
        selected_ages = st.sidebar.multiselect("Filter by poutcome:", age_options)

        # Data visualization
        st.write("Data Visualization:")
        fig = px.histogram(filtered_dataset, x='poutcome', title='poutcome Distribution')
        st.plotly_chart(fig)
    else:
        st.warning("Column 'poutcome' not found in the dataset.")

with col2:
    if 'housing' in dataset.columns:
        age_options = dataset['housing'].unique()
        selected_ages = st.sidebar.multiselect("Filter by housing:", age_options)
        
        # Data visualization
        st.write("Data Visualization:")
        fig = px.histogram(filtered_dataset, x='housing', title='housing Distribution')
        st.plotly_chart(fig)
    else:
        st.warning("Column 'housing' not found in the dataset.")



    if 'loan' in dataset.columns:
        age_options = dataset['loan'].unique()
        selected_ages = st.sidebar.multiselect("Filter by loan:", age_options)
        # Data visualization
        st.write("Data Visualization:")
        fig = px.histogram(filtered_dataset, x='loan', title='loan Distribution')
        st.plotly_chart(fig)
    else:
        st.warning("Column 'loan' not found in the dataset.")


    if 'contact' in dataset.columns:
        age_options = dataset['contact'].unique()
        selected_ages = st.sidebar.multiselect("Filter by Contact:", age_options)

        # Data visualization
        st.write("Data Visualization:")
        fig = px.histogram(filtered_dataset, x='contact', title='contact Distribution')
        st.plotly_chart(fig)
    else:
        st.warning("Column 'contact' not found in the dataset.")

    if 'day' in dataset.columns:
        age_options = dataset['day'].unique()
        selected_ages = st.sidebar.multiselect("Filter by day:", age_options)

        # Data visualization
        st.write("Data Visualization:")
        fig = px.histogram(filtered_dataset, x='day', title='day Distribution')
        st.plotly_chart(fig)
    else:
        st.warning("Column 'day' not found in the dataset.")


    if 'month' in dataset.columns:
        age_options = dataset['month'].unique()
        selected_ages = st.sidebar.multiselect("Filter by month:", age_options)

        # Data visualization
        st.write("Data Visualization:")
        fig = px.histogram(filtered_dataset, x='month', title='month Distribution')
        st.plotly_chart(fig)
    else:
        st.warning("Column 'month' not found in the dataset.")


    if 'duration' in dataset.columns:
        age_options = dataset['duration'].unique()
        selected_ages = st.sidebar.multiselect("Filter by duration:", age_options)

        # Data visualization
        st.write("Data Visualization:")
        fig = px.histogram(filtered_dataset, x='duration', title='duration Distribution')
        st.plotly_chart(fig)
    else:
        st.warning("Column 'duration' not found in the dataset.")

    if 'campaign' in dataset.columns:
        age_options = dataset['campaign'].unique()
        selected_ages = st.sidebar.multiselect("Filter by campaign:", age_options)

        # Data visualization
        st.write("Data Visualization:")
        fig = px.histogram(filtered_dataset, x='campaign', title='campaign Distribution')
        st.plotly_chart(fig)
    else:
        st.warning("Column 'campaign' not found in the dataset.")

    if 'pdays' in dataset.columns:
        age_options = dataset['pdays'].unique()
        selected_ages = st.sidebar.multiselect("Filter by pdays:", age_options)

        # Data visualization
        st.write("Data Visualization:")
        fig = px.histogram(filtered_dataset, x='pdays', title='pdays Distribution')
        st.plotly_chart(fig)
    else:
        st.warning("Column 'pdays' not found in the dataset.")



if 'Target' in dataset.columns:
    age_options = dataset['Target'].unique()
    selected_ages = st.sidebar.multiselect("Filter by Target:", age_options)

        # Data visualization
    st.write("Data Visualization:")
    fig = px.histogram(filtered_dataset, x='Target', title='Target Distribution')
    st.plotly_chart(fig)
else:
    st.warning("Column 'Target' not found in the dataset.")


st.title("Relationship between 'Balance' and 'Job'")
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.boxplot(data=dataset, x='job', y='balance')
plt.xticks(rotation=45, ha='right')
plt.xlabel("Job")
plt.ylabel("Balance")

# Display the plot in Streamlit
st.pyplot(plt)


if 'Target' in dataset.columns:
    target_options = dataset['Target'].unique()
    selected_targets = st.sidebar.multiselect("Select Target Values:", target_options)
    target_filtered_dataset = dataset[dataset['Target'].isin(selected_targets)]

    # Visualize relationship between target and job using subplots
    st.title("Relationship between Target and Job")

    # Get unique job categories
    job_categories = dataset['job'].unique()

    # Create a grid of subplots
    num_rows = (len(job_categories) + 1) // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows * 6))
    axes = axes.flatten()

    for i, job_category in enumerate(job_categories):
        ax = axes[i]
        job_subset = target_filtered_dataset[target_filtered_dataset['job'] == job_category]

        # Check if there is enough data for the boxplot
        if len(job_subset) > 0:
            sns.boxplot(data=job_subset, x='Target', y='age', ax=ax)
            ax.set_title(f"Job: {job_category}")
            ax.set_xlabel("Target")
            ax.set_ylabel("Age")
        else:
            ax.axis('off')  # Hide empty subplots

    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)

    st.title("Relationship between Target and Contact")

    # Get unique contact categories
    contact_categories = dataset['contact'].unique()

    # Create a grid of subplots
    num_rows = (len(contact_categories) + 1) // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows * 6))
    axes = axes.flatten()

    for i, contact_category in enumerate(contact_categories):
        ax = axes[i]
        contact_subset = target_filtered_dataset[target_filtered_dataset['contact'] == contact_category]

        # Check if there is enough data for the boxplot
        if len(contact_subset) > 0:
            sns.countplot(data=contact_subset, x='contact', hue='Target', ax=ax)
            ax.set_title(f"Contact: {contact_category}")
            ax.set_xlabel("Contact")
            ax.set_ylabel("Count")
            ax.legend(title='Target')
        else:
            ax.axis('off')  # Hide empty subplots

    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)
    st.title("Relationship between Target and Marital")

    # Get unique marital statuses
    marital_statuses = dataset['marital'].unique()

    # Create a grid of subplots
    num_rows = (len(marital_statuses) + 1) // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows * 6))
    axes = axes.flatten()

    for i, marital_status in enumerate(marital_statuses):
        ax = axes[i]
        marital_subset = target_filtered_dataset[target_filtered_dataset['marital'] == marital_status]
        sns.countplot(data=marital_subset, x='Target', ax=ax)
        ax.set_title(f"Marital: {marital_status}")
        ax.set_xlabel("Target")
        ax.set_ylabel("Count")

    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)

else:
    st.warning("Column 'Target' not found in the dataset.")





