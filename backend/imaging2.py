import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# Load the imaging biomarkers data
imaging_data = pd.read_csv('imaging_biomarkers.csv')

# Preview the data
print("Data Preview:")
print(imaging_data.head())

# Check for missing values
print("Missing values per column:")
print(imaging_data.isnull().sum())

# Drop rows with missing values in the target ('DX')
imaging_data_cleaned = imaging_data.dropna(subset=['DX'])

# Impute missing values in features using mean imputation
imputer = SimpleImputer(strategy='mean')
imaging_data_cleaned[['FDG', 'AV45', 'PIB', 'FBB']] = imputer.fit_transform(imaging_data_cleaned[['FDG', 'AV45', 'PIB', 'FBB']])

# Check missing values after imputation
print("Missing values after cleaning and imputation:")
print(imaging_data_cleaned.isnull().sum())

# Resample to balance the target variable
dementia_data = imaging_data_cleaned[imaging_data_cleaned['DX'] == 'Dementia']
non_dementia_data = imaging_data_cleaned[imaging_data_cleaned['DX'].isin(['MCI', 'CN'])]
non_dementia_resampled = resample(non_dementia_data, replace=False, n_samples=len(dementia_data), random_state=42)

# Combine the resampled non-dementia data with dementia data
balanced_data = pd.concat([dementia_data, non_dementia_resampled])

# Encode the target variable: 1 for Dementia, 0 for MCI/CN
balanced_data['DX'] = balanced_data['DX'].apply(lambda x: 1 if x == 'Dementia' else 0)

# Define features (X) and target (y)
X = balanced_data[['FDG', 'AV45', 'PIB', 'FBB']]  # Feature columns
y = balanced_data['DX']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the XGBoost model
model_xgb = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)

# Train the model
model_xgb.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = model_xgb.predict(X_test_scaled)
y_pred_proba = model_xgb.predict_proba(X_test_scaled)[:, 1]

# Classification report and ROC-AUC score
print("Classification Report:")
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC-AUC Score:", roc_auc)

# # Cross-validation for robust evaluation
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv_scores = cross_val_score(model_xgb, X_train_scaled, y_train, cv=skf, scoring='roc_auc')
# print("Cross-Validation AUC Scores:", cv_scores)
# print("Mean CV AUC:", cv_scores.mean())

# # Feature importance visualization
# xgb.plot_importance(model_xgb)
# plt.title('Feature Importance')
# plt.show()

# # SHAP for interpretability
# explainer = shap.TreeExplainer(model_xgb)
# shap_values = explainer.shap_values(X_test_scaled)

# # SHAP summary plot
# shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns)
# Save the model to a file
import joblib
    
joblib.dump(model_xgb, 'xgboost_model.joblib')
print("Model saved as 'xgboost_model.joblib'")
