import pandas as pd
import streamlit as st

def load_iris_dataset():
    # Function to load the preprocessed dataset
    file_path = 'C:/Users/aswin/InsightfulClusters/data/Iris.xlsx'
    dataset = pd.read_excel(file_path)
    return dataset

def load_wine_dataset():
    # Function to load the preprocessed dataset
    # Replace with your code to load the dataset
    file_path = 'C:/Users/aswin/InsightfulClusters/data/Wine.xlsx'
    dataset = pd.read_excel(file_path)
    return dataset

def load_mall_customer_dataset():
    # Function to load the preprocessed dataset
    file_path = 'C:/Users/aswin/InsightfulClusters/data/Mall_Customers.xlsx'
    dataset = pd.read_excel(file_path)
    return dataset

def load_diabetes_dataset():
    # Function to load the preprocessed dataset
    file_path = 'C:/Users/aswin/InsightfulClusters/data/Diabetes.xlsx'
    dataset = pd.read_excel(file_path)
    return dataset

def load_wild_blueberry_yield_dataset():
    # Function to load the preprocessed dataset
    file_path = 'C:/Users/aswin/InsightfulClusters/data/WildBlueberry_Yield.xlsx'
    dataset = pd.read_excel(file_path)
    return dataset

def load_sleep_health_lifestyle_dataset():
    # Function to load the preprocessed dataset
    file_path = 'C:/Users/aswin/InsightfulClusters/data/Sleep_Health_Lifestyle.xlsx'
    dataset = pd.read_excel(file_path)
    return dataset

def load_automobile_customer_segmentation_dataset():
    # Function to load the preprocessed dataset
    file_path = 'C:/Users/aswin/InsightfulClusters/data/Automobile_Customer_Segmentation.xlsx'
    dataset = pd.read_excel(file_path)
    return dataset

# Function to load the preprocessed dataset
def load_dataset(dataset_name):
    if dataset_name == 'Iris Flowers Clustering':
        # Load Iris dataset
        dataset = load_iris_dataset()
        return dataset
    elif dataset_name == 'Wine Varieties Clustering':
        # Load Wine dataset
        dataset = load_wine_dataset()
        return dataset
    elif dataset_name == 'Mall Shopper Segmentation':
        # Load Mall Customer dataset
        dataset = load_mall_customer_dataset()
        return dataset
    elif dataset_name == 'Diabetes Patient Clustering':
        # Load Mall Customer dataset
        dataset = load_diabetes_dataset()
        return dataset
    elif dataset_name == 'Blueberry Growth Characteristics':
        # Load Mall Customer dataset
        dataset = load_wild_blueberry_yield_dataset()
        return dataset
    elif dataset_name == 'Sleep, Health & Lifestyle':
        # Load Mall Customer dataset
        dataset = load_sleep_health_lifestyle_dataset()
        return dataset
    elif dataset_name == 'Automobile Customer Segmentation':
        # Load Mall Customer dataset
        dataset = load_automobile_customer_segmentation_dataset()
        return dataset
    elif dataset_name == 'Custom':
        # Custom dataset upload
        uploaded_file = st.file_uploader('Upload your dataset (Excel format)', type='xlsx')
        if uploaded_file is not None:
            # Read the uploaded Excel file
            dataset = pd.read_excel(uploaded_file)
            return dataset


