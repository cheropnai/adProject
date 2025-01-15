import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as imPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib  # Import joblib to save the model

# Load the demographic data
demographic_data = pd.read_csv('demographic_data.csv')

# Display the first few rows to verify the data
print("First few rows of Demographic Data:")
print(demographic_data.head())

# Check for basic information
print("\nDemographic Data Information:")
print(demographic_data.info())

# Check for missing values
print("\nMissing Values in Demographic Data:")
print(demographic_data.isnull().sum())

# Fill missing values for categorical columns (mode)
for column in demographic_data.select_dtypes(include=['object']).columns:
    mode_value = demographic_data[column].mode()[0]
    demographic_data[column] = demographic_data[column].fillna(mode_value)

# Replace missing values in AGE with the median
demographic_data['AGE'].fillna(demographic_data['AGE'].median(), inplace=True)

# Replace missing values in PTMARRY with the mode
demographic_data['PTMARRY'].fillna(demographic_data['PTMARRY'].mode()[0], inplace=True)

# Verify there are no missing values
print(demographic_data.isnull().sum())

# Map DX to binary target variable
demographic_data['AD_Diagnosis'] = demographic_data['DX'].map({'CN': 0, 'MCI': 0, 'Dementia': 1})
 # Confirm the new target column
print(demographic_data['AD_Diagnosis'].value_counts())

# Drop rows with missing target values
demographic_data = demographic_data.dropna(subset=['AD_Diagnosis'])

# Define the features (X) and target (y)
target = 'AD_Diagnosis'

# Keep only 'AGE' for the features
X = demographic_data[['AGE']]  # Use only 'AGE' as the feature
y = demographic_data[target]

# Preprocessing steps: scaling the numerical feature (AGE)
preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), ['AGE'])]  # Only 'AGE' needs scaling
)

# Define the model pipeline with SMOTE and Random Forest Classifier
pipeline = imPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Print the classification report and ROC AUC score
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

# Save the trained model to a file
joblib.dump(pipeline, 'demographic_model.joblib')  # Save both model and scaler as part of the pipeline

print("Model saved as 'demographic_model.joblib'")
