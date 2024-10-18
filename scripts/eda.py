import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data function
def load_data(file_path):
    """
    Function to load CSV data into a DataFrame.
    :param file_path: str - path to the CSV file
    :return: pandas DataFrame
    """
    return pd.read_csv(file_path)

# Univariate analysis function
def univariate_analysis(df, column):
    """
    Function to perform univariate analysis for a specific column.
    :param df: pandas DataFrame
    :param column: str - column to analyze
    """
    # Plot the distribution of numerical data
    if pd.api.types.is_numeric_dtype(df[column]):
        plt.figure(figsize=(10, 5))
        sns.histplot(df[column], kde=True, bins=30)
        plt.title(f'Univariate Analysis: Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()
    
    # Plot the count of categorical data
    elif pd.api.types.is_categorical_dtype(df[column]) or df[column].dtype == 'object':
        plt.figure(figsize=(10, 5))
        sns.countplot(x=column, data=df, order=df[column].value_counts().index)
        plt.title(f'Univariate Analysis: Count of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

# Bivariate analysis function
def bivariate_analysis(df, column1, column2):
    """
    Function to perform bivariate analysis between two columns.
    :param df: pandas DataFrame
    :param column1: str - first column for the analysis
    :param column2: str - second column for the analysis
    """
    # Numerical vs Numerical - Scatter plot
    if pd.api.types.is_numeric_dtype(df[column1]) and pd.api.types.is_numeric_dtype(df[column2]):
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=column1, y=column2, data=df)
        plt.title(f'Bivariate Analysis: {column1} vs {column2}')
        plt.xlabel(column1)
        plt.ylabel(column2)
        plt.show()

    # Categorical vs Numerical - Box plot
    elif pd.api.types.is_categorical_dtype(df[column1]) or df[column1].dtype == 'object':
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=column1, y=column2, data=df)
        plt.title(f'Bivariate Analysis: {column1} vs {column2}')
        plt.xlabel(column1)
        plt.ylabel(column2)
        plt.xticks(rotation=45)
        plt.show()

    # Categorical vs Categorical - Heatmap of count (crosstab)
    elif pd.api.types.is_categorical_dtype(df[column1]) and pd.api.types.is_categorical_dtype(df[column2]):
        cross_tab = pd.crosstab(df[column1], df[column2])
        plt.figure(figsize=(10, 5))
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlGnBu')
        plt.title(f'Bivariate Analysis: {column1} vs {column2}')
        plt.show()

# Exploratory Data Analysis function for Fraud data
def eda_fraud_data(fraud_df):
    """
    Function to perform exploratory data analysis (EDA) on the fraud dataset.
    :param fraud_df: pandas DataFrame - fraud dataset
    """
    print("Univariate Analysis:")
    univariate_analysis(fraud_df, 'purchase_value')
    univariate_analysis(fraud_df, 'source')

    print("Bivariate Analysis:")
    bivariate_analysis(fraud_df, 'source', 'purchase_value')
    bivariate_analysis(fraud_df, 'class', 'purchase_value')

# Exploratory Data Analysis function for IP address data
def eda_ip_data(ip_df):
    """
    Function to perform exploratory data analysis (EDA) on the IP address dataset.
    :param ip_df: pandas DataFrame - IP address dataset
    """
    print("Univariate Analysis:")
    univariate_analysis(ip_df, 'lower_bound_ip_address')
    univariate_analysis(ip_df, 'country')

    print("Bivariate Analysis:")
    bivariate_analysis(ip_df, 'lower_bound_ip_address', 'upper_bound_ip_address')

# Exploratory Data Analysis function for Credit card data
def eda_credit_card_data(cc_df):
    """
    Function to perform exploratory data analysis (EDA) on the credit card dataset.
    :param cc_df: pandas DataFrame - credit card dataset
    """
    print("Univariate Analysis:")
    univariate_analysis(cc_df, 'Amount')
    univariate_analysis(cc_df, 'Class')

    print("Bivariate Analysis:")
    bivariate_analysis(cc_df, 'Amount', 'Time')
    bivariate_analysis(cc_df, 'Class', 'Amount')

# Main function to load, clean, and perform EDA on datasets
def eda(fraud_df, ip_df, creditcard_df):
    
    # Perform EDA on datasets
    print("\n----- Fraud Data EDA -----")
    eda_fraud_data(fraud_df)
    
    print("\n----- IP Data EDA -----")
    eda_ip_data(ip_df)
    
    print("\n----- Credit Card Data EDA -----")
    eda_credit_card_data(creditcard_df)



