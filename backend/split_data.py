import pandas as pd

# Load your dataset (replace 'your_dataset.csv' with the path to your file)
df = pd.read_csv('dataset/ADNIMERGE_10Sep2024.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Show basic information about the dataset (number of rows, columns, data types, etc.)
print("\nDataset Info:")
print(df.info())

# Display descriptive statistics for numerical columns
print("\nDescriptive Statistics for Numerical Columns:")
print(df.describe())

# Show the number of missing values for each column
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Display the number of unique values for each column
print("\nUnique Values in Each Column:")
print(df.nunique())

# If there are categorical columns, display their unique values
categorical_columns = df.select_dtypes(include=['object']).columns
if len(categorical_columns) > 0:
    print("\nUnique Values in Categorical Columns:")
    for col in categorical_columns:
        print(f"{col}: {df[col].unique()}")

# Display correlation matrix for numerical features
print("\nCorrelation Matrix:")
numerical_columns = df.select_dtypes(include=['number'])  # Select only numerical columns
print(numerical_columns.corr())



# Define columns for each modality
demographic_columns = ['AGE', 'PTGENDER', 'PTETHCAT', 'PTRACCAT', 'PTMARRY']  # Demographic columns
clinical_columns = ['DX_bl', 'CDRSB', 'EcogSPDivatt_bl', 'EcogSPTotal_bl']  # Clinical columns
imaging_biomarkers_columns = ['FDG', 'AV45', 'PIB', 'FBB']  # Imaging biomarkers (e.g., PET scan data)
genetic_biomarkers_columns = ['APOE4', 'ABETA', 'TAU', 'PTAU', 'ABETA_bl', 'TAU_bl', 'PTAU_bl']  # Genetic biomarkers

# Ensure the Patient ID column is included in each split
patient_id_column = 'PTID'

# Check if PTID exists in your dataframe
if patient_id_column in df.columns:
    # Define columns for each modality, including Patient ID
    demographic_columns = ['PTID', 'AGE', 'PTGENDER', 'PTETHCAT', 'PTRACCAT', 'PTMARRY', 'DX']
    clinical_columns = ['PTID', 'DX_bl', 'CDRSB', 'EcogSPDivatt_bl', 'EcogSPTotal_bl', 'DX']
    imaging_biomarkers_columns = ['PTID', 'FDG', 'AV45', 'PIB', 'FBB', 'DX']
    genetic_biomarkers_columns = ['PTID', 'APOE4', 'ABETA', 'TAU', 'PTAU', 'ABETA_bl', 'TAU_bl', 'PTAU_bl', 'DX']

    # Split the dataset based on modality
    demographic_data = df[demographic_columns]
    clinical_data = df[clinical_columns]
    imaging_biomarkers = df[imaging_biomarkers_columns]
    genetic_biomarkers = df[genetic_biomarkers_columns]

    # Export each modality DataFrame to a separate CSV file
    demographic_data.to_csv('demographic_data.csv', index=False)
    clinical_data.to_csv('clinical_data.csv', index=False)
    imaging_biomarkers.to_csv('imaging_biomarkers.csv', index=False)
    genetic_biomarkers.to_csv('genetic_biomarkers.csv', index=False)

    print("Data has been successfully split and saved as separate files with Patient ID included.")
else:
    print(f"Column '{patient_id_column}' not found in the dataset.")
