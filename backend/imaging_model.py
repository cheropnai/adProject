import pandas as pd

# Load the imaging biomarkers data
imaging_data = pd.read_csv('imaging_biomarkers.csv')

# Preview the data
print("Data Preview:")
print(imaging_data.head())

# Check for missing values
print("Missing values per column:")
print(imaging_data.isnull().sum())
# Define features and target
X_imaging = imaging_data[['FDG', 'AV45', 'PIB', 'FBB']]
y_imaging = imaging_data['DX']

# Print the shape of the features and target
print(f"Features shape: {X_imaging.shape}")
print(f"Target shape: {y_imaging.shape}")
# Drop the 'PIB' feature
imaging_data = imaging_data.drop(columns=['PIB'])

# Drop rows with missing values in 'DX' (target variable)
imaging_data_cleaned = imaging_data.dropna(subset=['DX'])

# Check if there are any remaining missing values
print("Missing values per column after cleaning:")
print(imaging_data_cleaned.isnull().sum())

# Preview the cleaned data
print("Cleaned Data Preview:")
print(imaging_data_cleaned.head())
# Impute missing values with the mean of each column
imaging_data_cleaned['FDG'] = imaging_data_cleaned['FDG'].fillna(imaging_data_cleaned['FDG'].mean())
imaging_data_cleaned['AV45'] = imaging_data_cleaned['AV45'].fillna(imaging_data_cleaned['AV45'].mean())
imaging_data_cleaned['FBB'] = imaging_data_cleaned['FBB'].fillna(imaging_data_cleaned['FBB'].mean())

# Verify if there are any missing values left
print("Missing values per column after imputation:")
print(imaging_data_cleaned.isnull().sum())
# Define features and target after cleaning
X_imaging = imaging_data_cleaned[['FDG', 'AV45', 'FBB']]  # Excluding 'PIB' from the features
y_imaging = imaging_data_cleaned['DX']  # Target variable 'DX'

# Print the shape of the features and target
print(f"Features shape: {X_imaging.shape}")
print(f"Target shape: {y_imaging.shape}")
import pandas as pd
from sklearn.utils import resample

# Assuming 'imaging_data_cleaned' is your dataframe and 'DX' is the target column
# Step 1: Filter the 'Dementia' class and keep it as is
dementia_data = imaging_data_cleaned[imaging_data_cleaned['DX'] == 'Dementia']

# Step 2: Combine 'MCI' and 'CN' classes
non_dementia_data = imaging_data_cleaned[imaging_data_cleaned['DX'].isin(['MCI', 'CN'])]

# Step 3: Resample non-dementia class to have the same number of samples as 'Dementia' class
non_dementia_resampled = resample(non_dementia_data,
                                  replace=False,    # No replacement, we are reducing the samples
                                  n_samples=len(dementia_data),    # Equal to the number of 'Dementia' samples
                                  random_state=42)

# Step 4: Concatenate the resampled 'MCI' + 'CN' data with 'Dementia' data
balanced_data = pd.concat([dementia_data, non_dementia_resampled])

# Step 5: Update the target variable to 0 for the combined 'MCI' + 'CN' data
balanced_data['DX'] = balanced_data['DX'].apply(lambda x: 1 if x == 'Dementia' else 0)

# Check the new distribution of classes
print(balanced_data['DX'].value_counts())

import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Assuming you already have the balanced dataset 'balanced_data' from earlier
X = balanced_data.drop(columns=['DX'])  # Features (drop target column)
y = balanced_data['DX']  # Target variable

# Step 1: Identify non-numeric columns
non_numeric_cols = X.select_dtypes(exclude=['number']).columns
print("Non-numeric columns:", non_numeric_cols)

# Step 2: If there are non-numeric columns, drop them (or encode them if needed)
X = X.drop(columns=non_numeric_cols)  # Drop non-numeric columns for now

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the XGBoost model
model_xgb = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

# Train the model
model_xgb.fit(X_train_scaled, y_train)

# Predict the test set
y_pred = model_xgb.predict(X_test_scaled)

# Print the classification report
print(classification_report(y_test, y_pred))

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, model_xgb.predict_proba(X_test_scaled)[:, 1])
print("ROC-AUC Score:", roc_auc)
