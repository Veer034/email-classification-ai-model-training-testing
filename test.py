import pandas as pd
import os

# List of CSV files to process
csv_files = [
    'digital_healthcare_combined.csv',
    'ecommerce_combined.csv',
    'health_insurance_combined.csv',
    'hospitality_and_accommodation_combined.csv',
    'hospitals_and_healthcare_combined.csv',
    'it_combined.csv',
    'retail_combined.csv',
    'tourism_services_combined.csv',
    'travel_and_transportation_combined.csv'
]

# Initialize an empty list to store DataFrames
dataframes = []

# Read each CSV file and append to the list
for file in csv_files:
    file_path = os.path.join('/Users/ranveersingh/Desktop/Own/AI/combined-training-data', file)  # Replace 'path_to_your_folder' with the actual path
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        dataframes.append(df)
    else:
        print(f"File not found: {file_path}")

# Combine all DataFrames into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Select relevant columns and drop rows with missing department, type, or sub_type
columns_to_keep = ['department', 'type', 'sub_type']
final_df = combined_df[columns_to_keep].dropna()

# Remove duplicate combinations of department, type, and sub_type
unique_combinations_df = final_df.drop_duplicates(subset=['department', 'type', 'sub_type'])

# Ensure that only valid types are included
valid_types = ['query', 'complaint', 'suggestion']
unique_combinations_df = unique_combinations_df[unique_combinations_df['type'].isin(valid_types)]

# Sort the DataFrame by department, type, and sub_type
unique_combinations_df = unique_combinations_df.sort_values(['department', 'type', 'sub_type'])

# Save the unique combinations to a CSV file
output_file = 'unique_combinations.csv'
unique_combinations_df.to_csv(output_file, index=False)

# Output summary statistics
print(f"Unique combinations saved to {output_file}")
print(f"Total unique combinations: {len(unique_combinations_df)}")

# Display the first few rows
print("\nFirst few rows of the unique combinations:")
print(unique_combinations_df.head())

# Display some statistics
print("\nStatistics:")
print(f"Number of unique departments: {unique_combinations_df['department'].nunique()}")
print(f"Number of unique types: {unique_combinations_df['type'].nunique()}")
print(f"Number of unique sub-types: {unique_combinations_df['sub_type'].nunique()}")
