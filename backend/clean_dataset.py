import pandas as pd

# Load datasets
genetic_data = pd.read_csv('genetic_biomarkers.csv')
demographic_data = pd.read_csv('demographic_data.csv')
imaging_data = pd.read_csv('imaging_biomarkers.csv')

# Preview datasets
print("Genetic Data Preview:")
print(genetic_data.head())
print("\nDemographic Data Preview:")
print(demographic_data.head())
print("\nImaging Data Preview:")
print(imaging_data.head())
# Drop unnecessary columns
genetic_data = genetic_data.drop(columns=['ABETA', 'TAU', 'PTAU'], errors='ignore')

# Drop rows with missing values in critical columns
genetic_data = genetic_data.dropna(subset=['APOE4', 'ABETA_bl', 'TAU_bl', 'PTAU_bl', 'DX'])

# Encode target variable
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
genetic_data['DX'] = label_encoder.fit_transform(genetic_data['DX'])

print("Cleaned Genetic Data Preview:")
print(genetic_data.head())
# Fill missing AGE with median
demographic_data['AGE'].fillna(demographic_data['AGE'].median(), inplace=True)

# Drop rows with missing DX
demographic_data = demographic_data.dropna(subset=['DX'])

# Encode target variable
demographic_data['DX'] = label_encoder.fit_transform(demographic_data['DX'])

print("Cleaned Demographic Data Preview:")
print(demographic_data.head())
# Drop rows with missing target DX
imaging_data = imaging_data.dropna(subset=['DX'])

# Impute missing values for features
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
imaging_data[['FDG', 'AV45', 'PIB', 'FBB']] = imputer.fit_transform(imaging_data[['FDG', 'AV45', 'PIB', 'FBB']])

# Encode target variable
imaging_data['DX'] = label_encoder.fit_transform(imaging_data['DX'])

print("Cleaned Imaging Data Preview:")
print(imaging_data.head())
# Merge datasets on PTID
merged_data = genetic_data.merge(demographic_data, on='PTID').merge(imaging_data, on='PTID')

print("Merged Dataset Preview:")
print(merged_data.head())

# Define features for each modality
X_genetic = merged_data[['APOE4', 'ABETA_bl', 'TAU_bl', 'PTAU_bl']]  # Genetic features
X_demographic = merged_data[['AGE']]  # Demographic features
X_imaging = merged_data[['FDG', 'AV45', 'PIB', 'FBB']]  # Imaging features

# Define the target
y_combined = merged_data['DX']  # Target column

# Verify alignment
assert len(X_genetic) == len(X_demographic) == len(X_imaging) == len(y_combined), "Datasets are not aligned!"

# Print the shapes of features and labels
print(f"Genetic Features Shape: {X_genetic.shape}")
print(f"Demographic Features Shape: {X_demographic.shape}")
print(f"Imaging Features Shape: {X_imaging.shape}")
print(f"Target Shape: {y_combined.shape}")

# Preview the features and labels
print("\nSample Genetic Features:")
print(X_genetic.head())
print("\nSample Demographic Features:")
print(X_demographic.head())
print("\nSample Imaging Features:")
print(X_imaging.head())
print("\nSample Target Labels:")
print(y_combined.head())
# Save the cleaned and merged datasets
merged_data.to_csv('cleaned_merged_data.csv', index=False)
X_genetic.to_csv('genetic_features.csv', index=False)
X_demographic.to_csv('demographic_features.csv', index=False)
X_imaging.to_csv('imaging_features.csv', index=False)
y_combined.to_csv('target_labels.csv', index=False)

print("Cleaned and aligned datasets have been saved:")
print("- Full merged dataset: 'cleaned_merged_data.csv'")
print("- Genetic features: 'genetic_features.csv'")
print("- Demographic features: 'demographic_features.csv'")
print("- Imaging features: 'imaging_features.csv'")
print("- Target labels: 'target_labels.csv'")
