import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Load the genetic biomarkers data
genetic_data = pd.read_csv('genetic_biomarkers.csv')

# Preview the data
print("Data Preview:")
print(genetic_data.head())

# Drop the specified columns
genetic_data_drop = genetic_data.drop(columns=['ABETA', 'TAU', 'PTAU'])

# Drop rows where any of the specified columns have missing values
columns_to_check = ['APOE4', 'ABETA_bl', 'TAU_bl', 'PTAU_bl', 'DX']
genetic_data_cleaned = genetic_data_drop.dropna(subset=columns_to_check)

# Preview the cleaned data
print(f"Remaining rows after dropping missing values: {genetic_data_cleaned.shape[0]}")
print(genetic_data_cleaned.head())

# Encode 'DX' (target variable)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
genetic_data_cleaned['DX'] = label_encoder.fit_transform(genetic_data_cleaned['DX'])

# Preview the encoded data
print("Encoded Data Preview:")
print(genetic_data_cleaned.head())

# Define your features (X) and target (y)
X = genetic_data_cleaned.drop(columns=['PTID', 'DX'])  # Dropping PTID (ID column) and DX (target)
y = genetic_data_cleaned['DX']  # Target variable

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preview the train-test split
print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

# Convert non-numeric columns to numeric (handle invalid values gracefully)
X_train['ABETA_bl'] = pd.to_numeric(X_train['ABETA_bl'], errors='coerce')
X_train['TAU_bl'] = pd.to_numeric(X_train['TAU_bl'], errors='coerce')
X_train['PTAU_bl'] = pd.to_numeric(X_train['PTAU_bl'], errors='coerce')

X_test['ABETA_bl'] = pd.to_numeric(X_test['ABETA_bl'], errors='coerce')
X_test['TAU_bl'] = pd.to_numeric(X_test['TAU_bl'], errors='coerce')
X_test['PTAU_bl'] = pd.to_numeric(X_test['PTAU_bl'], errors='coerce')

# Drop rows with missing values in both X_train and y_train to ensure alignment
X_train = X_train.dropna()
y_train = y_train[X_train.index]  # Make sure y_train has the same index after dropping rows

X_test = X_test.dropna()
y_test = y_test[X_test.index]  # Ensure y_test is aligned with X_test

# Preview the shapes after dropping rows
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Apply SMOTE resampling on the training set to address class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize the Random Forest Classifier with the best hyperparameters
best_rf = RandomForestClassifier(
    n_estimators=229,      # Best number of trees
    max_depth=25,          # Best max depth
    min_samples_split=3,   # Best min samples to split
    min_samples_leaf=1,    # Best min samples per leaf
    max_features=None,     # Best max features
    random_state=42,
    class_weight='balanced'  # Handling class imbalance
)

# Train the model
best_rf.fit(X_train_resampled, y_train_resampled)

# Evaluate the model on the test set
y_pred = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test)

# Print the classification report and ROC AUC score
print("Classification Report:")

print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score

# Compute ROC AUC score for multi-class classification (with 'ovr' strategy)
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

print(f"ROC AUC Score: {roc_auc}")

import joblib

# Save the trained model to a file
joblib.dump(best_rf, 'genetic_model.pkl')