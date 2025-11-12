# CE 49X - Lab 2: Soil Test Data Analysis

# Student Name: Eren Kurt
# Student ID: 2020403045
# Date: 16/10/2025

import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load the soil test dataset from a CSV file.
    
    Parameters:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded DataFrame, or None if the file is not found.
    """
    #Implement data loading with error handling
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from '{file_path}'.")
        return df
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
        return None
    pass

def clean_data(df):
    """
    Clean the dataset by handling missing values and removing outliers from 'soil_ph'.
    
    For each column in ['soil_ph', 'nitrogen', 'phosphorus', 'moisture']:
    - Missing values are filled with the column mean.
    
    Additionally, remove outliers in 'soil_ph' that are more than 3 standard deviations from the mean.
    
    Parameters:
        df (pd.DataFrame): The raw DataFrame.
        
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df_cleaned = df.copy()
    
    # Fill missing values in each specified column with the column mean
    columns_to_clean = ['soil_ph', 'nitrogen', 'phosphorus', 'moisture']
    for col in columns_to_clean:
        mean_val = df_cleaned[col].mean()
        df_cleaned[col].fillna(mean_val, inplace=True)
    
    # Remove outliers in 'soil_ph': values more than 3 standard deviations from the mean
    ph_mean = df_cleaned['soil_ph'].mean()
    ph_std = df_cleaned['soil_ph'].std()
    lower_bound = ph_mean - 3 * ph_std
    upper_bound = ph_mean + 3 * ph_std

    initial_rows = len(df_cleaned)
    df_cleaned = df_cleaned[(df_cleaned['soil_ph'] >= lower_bound) & (df_cleaned['soil_ph'] <= upper_bound)]
    rows_removed = initial_rows - len(df_cleaned)
    if rows_removed > 0:
        print(f"Removed {rows_removed} outliers from 'soil_ph'.")
#Displayed the preview with first 5 rows
    print(df_cleaned.head())
    return df_cleaned

def compute_statistics(df, column):
    """
    Compute and print descriptive statistics for the specified column.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column for which to compute statistics.
    """
    if column not in df.columns:
        print(f"Error: Column '{column}' not found in the DataFrame.")
        return
    # Calculate minimum value
    min_val = df[column].min()

    # Calculate maximum value
    max_val = df[column].max()

    #Calculate mean value
    mean_val = df[column].mean()

    #Calculate median value
    median_val = df[column].median()

    #Calculate standard deviation
    std_val = df[column].std()

    print(f"\nDescriptive statistics for '{column}':")
    print(f"  Minimum: {min_val}")
    print(f"  Maximum: {max_val}")
    print(f"  Mean: {mean_val:.2f}")
    print(f"  Median: {median_val:.2f}")
    print(f"  Standard Deviation: {std_val:.2f}")

def main():
    # Update the file path to point to your soil_test.csv file
    file_path = 'soil_test.csv'  # Update this path as needed
    
    # Load the dataset using the load_data function
    df_raw = load_data(file_path)

    # Clean the dataset using the clean_data function
    if df_raw is not None:
        #  Clean the dataset using the clean_data function
        df_clean = clean_data(df_raw)
    # Compute and display statistics for the 'soil_ph' column
    compute_statistics(df_clean, 'soil_ph')

    # Compute statistics for other columns
    compute_statistics(df_clean, 'nitrogen')
    compute_statistics(df_clean, 'phosphorus')
    compute_statistics(df_clean, 'moisture')
    
if __name__ == '__main__':
    main()

# =============================================================================
# REFLECTION QUESTIONS
# =============================================================================
# Answer these questions in comments below:

# 1. What was the most challenging part of this lab?
# Answer: The most challenging part was getting familiar with using modules.

# 2. How could soil data analysis help civil engineers in real projects?
# Answer: Soil data analysis helps civil engineers assess soil strength and stability, ensuring safe
# foundation design and durable construction.

# 3. What additional features would make this soil analysis tool more useful?
# Answer: Adding data visualization, automated reports, and comparison tools for different soil samples would make the tool more useful.

# 4. How did error handling improve the robustness of your code?
# Answer: Error handling improved the robustness of my code by preventing it from crashing when unexpected situations occurred,
# such as missing files, invalid column names, or data type issues. Instead of stopping execution, the program now displays
# clear error messages and continues safely, making it more reliable and user-friendly.