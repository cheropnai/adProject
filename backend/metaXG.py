import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

# Step 1: Load Processed Data
processed_data = pd.read_csv('processed_data.csv')

# Step 2: Split Data by Modality for Predictions
# Select features specific to each model
genetic_features = ['APOE4', 'ABETA_bl', 'TAU_bl', 'PTAU_bl']
demographic_features = ['AGE', 'PTGENDER', 'PTETHCAT', 'PTRACCAT', 'PTMARRY']
imaging_features = ['FDG', 'AV45', 'PIB', 'FBB']

# Split features and target
X = processed_data.drop(columns=['DX', 'PTID'])
y = processed_data['DX']

X_genetic = processed_data[genetic_features]
X_demographic = processed_data[demographic_features]
X_imaging = processed_data[imaging_features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_genetic = X_train[genetic_features]
X_train_demographic = X_train[demographic_features]
X_train_imaging = X_train[imaging_features]

X_test_genetic = X_test[genetic_features]
X_test_demographic = X_test[demographic_features]
X_test_imaging = X_test[imaging_features]

# Step 3: Load Pre-Trained Base Models
genetic_model = joblib.load('genetic_model.pkl')
demographic_model = joblib.load('demographic_model.joblib')
imaging_model = joblib.load('xgboost_model.joblib')

# Step 4: Get Predictions from Base Models
genetic_train_preds = genetic_model.predict_proba(X_train_genetic)[:, 1]
genetic_test_preds = genetic_model.predict_proba(X_test_genetic)[:, 1]

demographic_train_preds = demographic_model.predict_proba(X_train_demographic)[:, 1]
demographic_test_preds = demographic_model.predict_proba(X_test_demographic)[:, 1]

imaging_train_preds = imaging_model.predict_proba(X_train_imaging)[:, 1]
imaging_test_preds = imaging_model.predict_proba(X_test_imaging)[:, 1]

# Combine predictions to create meta-features
X_meta_train = np.column_stack([genetic_train_preds, demographic_train_preds, imaging_train_preds])
X_meta_test = np.column_stack([genetic_test_preds, demographic_test_preds, imaging_test_preds])

# Step 5: Train Meta-Learner (XGBoost)
meta_learner = xgb.XGBClassifier(
    objective='binary:logistic',  # Use binary classification
    eval_metric='logloss',
    n_estimators=100,  # Number of trees
    learning_rate=0.1,  # Step size shrinkage
    max_depth=5,  # Maximum depth of a tree
    subsample=0.8,  # Subsample ratio of training instances
    colsample_bytree=0.8,  # Subsample ratio of columns for each tree
    random_state=42
)

# Train the meta-learner
meta_learner.fit(X_meta_train, y_train)

# Step 6: Evaluate Meta-Learner
# Evaluate the Meta-Learner
y_meta_pred = meta_learner.predict(X_meta_test)
y_meta_pred_proba = meta_learner.predict_proba(X_meta_test)

# Print classification report
print("Classification Report for Meta-Learner:")
print(classification_report(y_test, y_meta_pred))

# Compute ROC AUC score for multi-class classification
roc_auc = roc_auc_score(y_test, y_meta_pred_proba, multi_class='ovr')
print(f"ROC AUC Score: {roc_auc}")


# Step 7: Save the Meta-Learner
joblib.dump(meta_learner, 'meta_learner_xgb.joblib')
print("Meta-Learner model saved as 'meta_learner_xgb.joblib'")
