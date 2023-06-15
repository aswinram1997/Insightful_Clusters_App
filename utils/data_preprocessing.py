import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from category_encoders import CountEncoder
from sklearn.preprocessing import LabelEncoder

def extract_column_types(dataset):
    """
    Extracts the numerical, categorical, and date columns from a dataset.

    Parameters:
    - dataset (pd.DataFrame): The dataset to extract column types from.

    Returns:
    - numerical_columns (list): List of numerical column names.
    - categorical_columns (list): List of categorical column names.
    - date_columns (list): List of date column names.
    """
    numerical_columns = []
    categorical_columns = []
    date_columns = []

    for column in dataset.columns:
        if dataset[column].dtype in ['int64', 'float64']:
            numerical_columns.append(column)
        elif dataset[column].dtype == 'object':
            categorical_columns.append(column)
        elif dataset[column].dtype == 'datetime64[ns]':
            date_columns.append(column)

    return numerical_columns, categorical_columns, date_columns


def handle_missing_values(dataset, numerical_columns, categorical_columns, date_columns):
    """
    Handles missing values in a pandas DataFrame for numerical, categorical, and date features.

    Parameters:
    - dataset (pd.DataFrame): The input DataFrame.
    - numerical_columns (list): List of numerical column names.
    - categorical_columns (list): List of categorical column names.
    - date_columns (list): List of date column names.

    Returns:
    - dataset (pd.DataFrame): The DataFrame with missing values handled.
    """

    # Handle missing values for numerical features
    if numerical_columns:
        dataset[numerical_columns] = dataset[numerical_columns].fillna(dataset[numerical_columns].mean())

    # Handle missing values for categorical features
    if categorical_columns:
        dataset[categorical_columns] = dataset[categorical_columns].fillna(dataset[categorical_columns].mode().iloc[0])

    # Handle missing values for date features
    if date_columns:
        dataset[date_columns] = dataset[date_columns].fillna(pd.NaT)

    return dataset


def handle_date_feature(dataset, date_columns):
    """
    Handles the date features in the dataset by extracting relevant components,
    discarding the original date columns, and performing label encoding on the
    engineered date columns.

    Parameters:
    - dataset (pd.DataFrame): The dataset containing the date features.
    - date_columns (list): A list of column names corresponding to the date features in the dataset.

    Returns:
    - updated_dataset (pd.DataFrame): The updated dataset with the original date columns dropped
                                      and the label-encoded date column.
    """

    for date_column in date_columns:
        # Perform label encoding on the date column
        label_encoder = LabelEncoder()
        dataset[date_column] = label_encoder.fit_transform(dataset[date_column])

    return dataset



def encode_categorical_features(dataset, categorical_columns):
    """
    Performs count encoding for high cardinality categorical features and one-hot encoding for other categorical features.

    Parameters:
    - dataset (pd.DataFrame): The dataset to be encoded.
    - categorical_columns (list): List of categorical column names to be encoded.

    Returns:
    - encoded_dataset (pd.DataFrame): The encoded dataset.
    - encoded_columns (list): List of column names after encoding.
    """
    if categorical_columns:
        high_cardinality_threshold = 10  # Define the threshold for high cardinality

        count_encoded_cols = []
        onehot_encoded_cols = []

        for column in categorical_columns:
            if dataset[column].nunique() > high_cardinality_threshold:
                count_encoder = CountEncoder()
                encoded_col = count_encoder.fit_transform(dataset[column])
                dataset[column] = encoded_col
                count_encoded_cols.append(column)
            else:
                onehot_encoder = OneHotEncoder(sparse=False, drop='first')
                encoded_cols = pd.DataFrame(onehot_encoder.fit_transform(dataset[[column]]))
                encoded_cols.columns = onehot_encoder.get_feature_names([column])
                dataset = pd.concat([dataset.drop(columns=[column]), encoded_cols], axis=1)
                onehot_encoded_cols.extend(encoded_cols.columns.tolist())

        encoded_columns = count_encoded_cols + onehot_encoded_cols
        
    else:
        encoded_columns = []
        
    return dataset, encoded_columns

def normalize_dataset(dataset, numerical_columns):
    """
    Perform normalization on numerical variables in a dataset.

    Parameters:
        dataset (pandas.DataFrame): The dataset to be normalized.
        numerical_columns (list): A list of column names that contain numerical variables.

    Returns:
        pandas.DataFrame: The modified dataset with the numerical columns normalized.
    """
    if numerical_columns:
        scaler = MinMaxScaler() 
        dataset[numerical_columns] = scaler.fit_transform(dataset[numerical_columns])  
        
    else: 
        numerical_columns = []
        
    return dataset 
