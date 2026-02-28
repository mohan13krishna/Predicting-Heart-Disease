"""
Heart Disease Prediction - Version 7 Ensemble Model
Advanced stacking with 5 base models + logistic regression meta-learner
Score: 0.95324 ROC-AUC
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

print("Training data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("\nTarget distribution:")
print(train_df['Heart Disease'].value_counts())

# Separate features and target
target_column = 'Heart Disease'
X_train = train_df.drop([target_column, 'id'], axis=1)
y_train = train_df[target_column]
X_test = test_df.drop('id', axis=1)
test_ids = test_df['id'].values

# Handle missing values
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Scale features
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

print("\nPreprocessed training shape:", X_train_scaled.shape)
print("Preprocessed test shape:", X_test_scaled.shape)

print("\nConverting target variable to numeric...")
y_train = y_train.map({'Absence': 0, 'Presence': 1})
print(f"Target converted. Unique values: {y_train.unique()}")

# Cross-validation setup
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Storage for OOF and test predictions
oof_preds_xgb = np.zeros(len(X_train_scaled))
test_preds_xgb = np.zeros(len(X_test_scaled))

oof_preds_lgb = np.zeros(len(X_train_scaled))
test_preds_lgb = np.zeros(len(X_test_scaled))

oof_preds_cat = np.zeros(len(X_train_scaled))
test_preds_cat = np.zeros(len(X_test_scaled))

oof_preds_rf = np.zeros(len(X_train_scaled))
test_preds_rf = np.zeros(len(X_test_scaled))

oof_preds_gb = np.zeros(len(X_train_scaled))
test_preds_gb = np.zeros(len(X_test_scaled))

cv_scores_xgb, cv_scores_lgb, cv_scores_cat, cv_scores_rf, cv_scores_gb = [], [], [], [], []

print("\nStarting 5-Fold Cross-Validation Training...\n")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
    print(f"Fold {fold + 1}/{n_folds}")

    X_fold_train = X_train_scaled.iloc[train_idx]
    X_fold_val = X_train_scaled.iloc[val_idx]
    y_fold_train = y_train.iloc[train_idx]
    y_fold_val = y_train.iloc[val_idx]

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, eval_metric='logloss'
    )
    xgb_model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)], verbose=False)
    oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_fold_val)[:, 1]
    test_preds_xgb += xgb_model.predict_proba(X_test_scaled)[:, 1] / n_folds
    cv_scores_xgb.append(roc_auc_score(y_fold_val, oof_preds_xgb[val_idx]))

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, verbose=-1
    )
    lgb_model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)])
    oof_preds_lgb[val_idx] = lgb_model.predict_proba(X_fold_val)[:, 1]
    test_preds_lgb += lgb_model.predict_proba(X_test_scaled)[:, 1] / n_folds
    cv_scores_lgb.append(roc_auc_score(y_fold_val, oof_preds_lgb[val_idx]))

    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=500, depth=7, learning_rate=0.05,
        random_state=42, verbose=False, thread_count=-1
    )
    cat_model.fit(X_fold_train, y_fold_train, eval_set=(X_fold_val, y_fold_val), verbose=False)
    oof_preds_cat[val_idx] = cat_model.predict_proba(X_fold_val)[:, 1]
    test_preds_cat += cat_model.predict_proba(X_test_scaled)[:, 1] / n_folds
    cv_scores_cat.append(roc_auc_score(y_fold_val, oof_preds_cat[val_idx]))

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=500, max_depth=15, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_fold_train, y_fold_train)
    oof_preds_rf[val_idx] = rf_model.predict_proba(X_fold_val)[:, 1]
    test_preds_rf += rf_model.predict_proba(X_test_scaled)[:, 1] / n_folds
    cv_scores_rf.append(roc_auc_score(y_fold_val, oof_preds_rf[val_idx]))

    # Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    gb_model.fit(X_fold_train, y_fold_train)
    oof_preds_gb[val_idx] = gb_model.predict_proba(X_fold_val)[:, 1]
    test_preds_gb += gb_model.predict_proba(X_test_scaled)[:, 1] / n_folds
    cv_scores_gb.append(roc_auc_score(y_fold_val, oof_preds_gb[val_idx]))

print("\n" + "="*50)
print("Cross-Validation Scores Summary:")
print("="*50)
print(f"XGBoost:          Mean AUC = {np.mean(cv_scores_xgb):.6f} (+/- {np.std(cv_scores_xgb):.6f})")
print(f"LightGBM:         Mean AUC = {np.mean(cv_scores_lgb):.6f} (+/- {np.std(cv_scores_lgb):.6f})")
print(f"CatBoost:         Mean AUC = {np.mean(cv_scores_cat):.6f} (+/- {np.std(cv_scores_cat):.6f})")
print(f"Random Forest:    Mean AUC = {np.mean(cv_scores_rf):.6f} (+/- {np.std(cv_scores_rf):.6f})")
print(f"Gradient Boosting:Mean AUC = {np.mean(cv_scores_gb):.6f} (+/- {np.std(cv_scores_gb):.6f})")

# Meta-model stacking
meta_train = pd.DataFrame({
    'xgb': oof_preds_xgb, 'lgb': oof_preds_lgb, 'cat': oof_preds_cat,
    'rf': oof_preds_rf, 'gb': oof_preds_gb
})
meta_test = pd.DataFrame({
    'xgb': test_preds_xgb, 'lgb': test_preds_lgb, 'cat': test_preds_cat,
    'rf': test_preds_rf, 'gb': test_preds_gb
})

meta_model = LogisticRegression(max_iter=1000, random_state=42)
meta_model.fit(meta_train, y_train)

final_train_preds = meta_model.predict_proba(meta_train)[:, 1]
final_test_preds = meta_model.predict_proba(meta_test)[:, 1]

print(f"\nMeta-model AUC Score: {roc_auc_score(y_train, final_train_preds):.6f}")

# Ensemble comparison
simple_avg = (oof_preds_xgb + oof_preds_lgb + oof_preds_cat + oof_preds_rf + oof_preds_gb) / 5
print(f"Simple Average AUC: {roc_auc_score(y_train, simple_avg):.6f}")

best_ensemble_test = final_test_preds

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'Heart Disease': best_ensemble_test
})
submission['Heart Disease'] = submission['Heart Disease'].clip(0, 1)
submission.to_csv('submission.csv', index=False)

print("\nSubmission file created successfully!")
print(submission.head(10))
print(f"\nPrediction range: [{submission['Heart Disease'].min():.6f}, {submission['Heart Disease'].max():.6f}]")
print("File saved as: submission.csv")
