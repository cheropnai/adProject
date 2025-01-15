import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Step 1: Load Data
merged_data = pd.read_csv('cleaned_merged_data.csv')

# Step 2: Identify Categorical Columns and Target Column
categorical_columns = ['PTGENDER', 'PTETHCAT', 'PTRACCAT', 'PTMARRY']
target_column = 'DX'

# Step 3: Encode Categorical Columns
label_encoder = LabelEncoder()

for col in categorical_columns:
    merged_data[col] = label_encoder.fit_transform(merged_data[col])

# Encode the target column
merged_data[target_column] = label_encoder.fit_transform(merged_data[target_column])

# Print Label Mapping for the Target Column
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping for DX:", label_mapping)

# Step 4: Check for Missing Values and Handle Them
print("Missing Values Before Cleaning:")
print(merged_data.isnull().sum())

# Drop rows with missing values
merged_data = merged_data.dropna()

print("Missing Values After Cleaning:")
print(merged_data.isnull().sum())

# Step 5: Split Dataset into Features (X) and Target (y)
X = merged_data.drop(columns=[target_column, 'PTID'])  # Drop target and ID columns
y = merged_data[target_column]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 6: **Fix Non-Numeric Values (Address the `'>1700'` Issue Here)**
for col in X_train.columns:
    if X_train[col].dtype == 'object':  # Check for non-numeric columns
        print(f"Unique values in {col}: {X_train[col].unique()}")
        X_train[col] = X_train[col].replace('>1700', 1700)  # Replace '>1700' with 1700
        X_test[col] = X_test[col].replace('>1700', 1700)    # Replace '>1700' with 1700 in test set

# Convert all columns to numeric
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# Handle missing values after conversion
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# Step 7: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 8: Handle Imbalance Using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Save Cleaned and Processed Data
merged_data.to_csv('processed_data.csv', index=False)
print("Processed dataset saved as 'processed_data.csv'.")

# Verify Shapes
print(f"X_train shape: {X_train_resampled.shape}")
print(f"y_train shape: {y_train_resampled.shape}")
print(f"X_test shape: {X_test_scaled.shape}")
print(f"y_test shape: {y_test.shape}")
