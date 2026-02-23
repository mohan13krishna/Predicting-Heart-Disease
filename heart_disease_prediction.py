"""
Predicting Heart Disease: Ensemble Learning Challenge
Kaggle Playground Series Season 6 Episode 2

This script implements an advanced ensemble learning strategy to predict 
heart disease using multiple models (XGBoost, LightGBM, CatBoost, Random Forest, Gradient Boosting)
combined through meta-model stacking.

Expected Score: ROC-AUC > 0.954
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

print("=" * 60)
print("PREDICTING HEART DISEASE - ENSEMBLE LEARNING")
print("=" * 60)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading datasets from Kaggle...")
train_df = pd.read_csv('/kaggle/input/playground-series-s6e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s6e2/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s6e2/sample_submission.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"\nFirst few rows of training data:")
print(train_df.head())
print(f"\nData types:\n{train_df.dtypes}")
print(f"\nMissing values:\n{train_df.isnull().sum()}")

# ============================================================================
# 2. DATA EXPLORATION
# ============================================================================
print("\n2. Data Exploration...")
print(f"\nTarget distribution:\n{train_df['Heart Disease'].value_counts()}")
print(f"\nClass balance:\n{train_df['Heart Disease'].value_counts(normalize=True)}")

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================
print("\n3. Preprocessing data...")

# Separate features and target
target_column = 'Heart Disease'
X_train = train_df.drop([target_column, 'id'], axis=1)
y_train = train_df[target_column]
X_test = test_df.drop('id', axis=1)
test_ids = test_df['id'].values

print(f"Features shape: {X_train.shape}")
print(f"Target shape: {y_train.shape}")

# Handle missing values
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Normalize features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print(f"Preprocessed training shape: {X_train_scaled.shape}")
print(f"Preprocessed test shape: {X_test_scaled.shape}")

# ============================================================================
# 4. CONVERT TARGET TO NUMERIC
# ============================================================================
print("\n4. Converting target variable to numeric...")
if y_train.dtype == 'object':
    print("Target is object type, converting...")
    unique_values = y_train.unique()
    print(f"Unique values: {unique_values}")
    
    if len(unique_values) == 2:
        y_train = pd.Series(y_train.map({unique_values[0]: 0, unique_values[1]: 1}).values)
    else:
        y_train = y_train.astype('category').cat.codes
    
    print(f"Conversion complete. New unique values: {y_train.unique()}")
else:
    print("Target is already numeric.")

print(f"Final target dtype: {y_train.dtype}")

# ============================================================================
# 5. K-FOLD CROSS-VALIDATION WITH MULTIPLE BASE MODELS
# ============================================================================
print("\n5. Training base models with 5-fold cross-validation...")
print("This may take 10-15 minutes on Kaggle GPU...\n")

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Initialize OOF and test predictions storage
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

cv_scores_xgb = []
cv_scores_lgb = []
cv_scores_cat = []
cv_scores_rf = []
cv_scores_gb = []

print("Starting 5-Fold Cross-Validation Training...\n")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
    print(f"Fold {fold + 1}/{n_folds}")
    
    X_fold_train = X_train_scaled.iloc[train_idx]
    X_fold_val = X_train_scaled.iloc[val_idx]
    y_fold_train = y_train.iloc[train_idx]
    y_fold_val = y_train.iloc[val_idx]
    
    # XGBoost
    print("  Training XGBoost...", end=" ")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, eval_metric='logloss'
    )
    xgb_model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)], verbose=False)
    oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_fold_val)[:, 1]
    test_preds_xgb += xgb_model.predict_proba(X_test_scaled)[:, 1] / n_folds
    cv_scores_xgb.append(roc_auc_score(y_fold_val, oof_preds_xgb[val_idx]))
    print(f"AUC: {cv_scores_xgb[-1]:.6f}")
    
    # LightGBM
    print("  Training LightGBM...", end=" ")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, verbose=-1
    )
    lgb_model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)])
    oof_preds_lgb[val_idx] = lgb_model.predict_proba(X_fold_val)[:, 1]
    test_preds_lgb += lgb_model.predict_proba(X_test_scaled)[:, 1] / n_folds
    cv_scores_lgb.append(roc_auc_score(y_fold_val, oof_preds_lgb[val_idx]))
    print(f"AUC: {cv_scores_lgb[-1]:.6f}")
    
    # CatBoost
    print("  Training CatBoost...", end=" ")
    cat_model = CatBoostClassifier(
        iterations=500, depth=7, learning_rate=0.05, random_state=42,
        verbose=False, thread_count=-1
    )
    cat_model.fit(X_fold_train, y_fold_train, eval_set=(X_fold_val, y_fold_val), verbose=False)
    oof_preds_cat[val_idx] = cat_model.predict_proba(X_fold_val)[:, 1]
    test_preds_cat += cat_model.predict_proba(X_test_scaled)[:, 1] / n_folds
    cv_scores_cat.append(roc_auc_score(y_fold_val, oof_preds_cat[val_idx]))
    print(f"AUC: {cv_scores_cat[-1]:.6f}")
    
    # Random Forest
    print("  Training Random Forest...", end=" ")
    rf_model = RandomForestClassifier(
        n_estimators=500, max_depth=15, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_fold_train, y_fold_train)
    oof_preds_rf[val_idx] = rf_model.predict_proba(X_fold_val)[:, 1]
    test_preds_rf += rf_model.predict_proba(X_test_scaled)[:, 1] / n_folds
    cv_scores_rf.append(roc_auc_score(y_fold_val, oof_preds_rf[val_idx]))
    print(f"AUC: {cv_scores_rf[-1]:.6f}")
    
    # Gradient Boosting
    print("  Training Gradient Boosting...", end=" ")
    gb_model = GradientBoostingClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    gb_model.fit(X_fold_train, y_fold_train)
    oof_preds_gb[val_idx] = gb_model.predict_proba(X_fold_val)[:, 1]
    test_preds_gb += gb_model.predict_proba(X_test_scaled)[:, 1] / n_folds
    cv_scores_gb.append(roc_auc_score(y_fold_val, oof_preds_gb[val_idx]))
    print(f"AUC: {cv_scores_gb[-1]:.6f}\n")

# Print CV scores summary
print("\n" + "=" * 60)
print("CROSS-VALIDATION SCORES SUMMARY")
print("=" * 60)
print(f"XGBoost:         Mean AUC = {np.mean(cv_scores_xgb):.6f} (+/- {np.std(cv_scores_xgb):.6f})")
print(f"LightGBM:        Mean AUC = {np.mean(cv_scores_lgb):.6f} (+/- {np.std(cv_scores_lgb):.6f})")
print(f"CatBoost:        Mean AUC = {np.mean(cv_scores_cat):.6f} (+/- {np.std(cv_scores_cat):.6f})")
print(f"Random Forest:   Mean AUC = {np.mean(cv_scores_rf):.6f} (+/- {np.std(cv_scores_rf):.6f})")
print(f"Gradient Boost:  Mean AUC = {np.mean(cv_scores_gb):.6f} (+/- {np.std(cv_scores_gb):.6f})")

# ============================================================================
# 6. ENSEMBLE & META-MODEL STACKING
# ============================================================================
print("\n6. Creating ensemble with meta-model stacking...")

# Create meta-features
meta_train = pd.DataFrame({
    'xgb': oof_preds_xgb, 'lgb': oof_preds_lgb, 'cat': oof_preds_cat,
    'rf': oof_preds_rf, 'gb': oof_preds_gb
})

meta_test = pd.DataFrame({
    'xgb': test_preds_xgb, 'lgb': test_preds_lgb, 'cat': test_preds_cat,
    'rf': test_preds_rf, 'gb': test_preds_gb
})

print(f"Meta-train shape: {meta_train.shape}")
print(f"Meta-test shape: {meta_test.shape}")

# Train meta-model
meta_model = LogisticRegression(max_iter=1000, random_state=42)
meta_model.fit(meta_train, y_train)

final_train_preds = meta_model.predict_proba(meta_train)[:, 1]
final_test_preds = meta_model.predict_proba(meta_test)[:, 1]

print(f"\nMeta-model AUC Score: {roc_auc_score(y_train, final_train_preds):.6f}")

# Compare ensemble methods
print("\n" + "=" * 60)
print("ENSEMBLE METHODS COMPARISON")
print("=" * 60)

simple_avg = (oof_preds_xgb + oof_preds_lgb + oof_preds_cat + oof_preds_rf + oof_preds_gb) / 5
print(f"Simple Average AUC:        {roc_auc_score(y_train, simple_avg):.6f}")

weights_cv = np.array([np.mean(cv_scores_xgb), np.mean(cv_scores_lgb), 
                        np.mean(cv_scores_cat), np.mean(cv_scores_rf), 
                        np.mean(cv_scores_gb)])
weights_cv = weights_cv / weights_cv.sum()
weighted_avg = (weights_cv[0]*oof_preds_xgb + weights_cv[1]*oof_preds_lgb + 
                weights_cv[2]*oof_preds_cat + weights_cv[3]*oof_preds_rf + 
                weights_cv[4]*oof_preds_gb)
print(f"Weighted Average AUC:      {roc_auc_score(y_train, weighted_avg):.6f}")
print(f"Meta-Model (Stacking) AUC: {roc_auc_score(y_train, final_train_preds):.6f}")

# ============================================================================
# 7. CREATE SUBMISSION
# ============================================================================
print("\n7. Creating submission file...")

submission = pd.DataFrame({
    'id': test_ids,
    'Heart Disease': final_test_preds
})

submission['Heart Disease'] = submission['Heart Disease'].clip(0, 1)
submission.to_csv('/kaggle/working/submission.csv', index=False)

print("✓ Submission file created successfully!")
print(f"\nFirst 10 rows of submission:")
print(submission.head(10))
print(f"\nSubmission shape: {submission.shape}")
print(f"Prediction range: [{submission['Heart Disease'].min():.6f}, {submission['Heart Disease'].max():.6f}]")
print(f"\nFile saved as: /kaggle/working/submission.csv")

print("\n" + "=" * 60)
print("PROCESS COMPLETE - Ready for Kaggle Submission!")
print("=" * 60)
