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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Separate features and target
target_column = 'Heart Disease'
X_train = train_df.drop([target_column, 'id'], axis=1)
y_train = train_df[target_column].map({'Absence': 0, 'Presence': 1})
X_test = test_df.drop('id', axis=1)
test_ids = test_df['id'].values

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Target distribution:\n{y_train.value_counts()}")

#  Feature Engineering
def add_features(df):
    df = df.copy()
    df['Age_MaxHR'] = df['Age'] * df['Max HR']
    df['Age_BP'] = df['Age'] * df['BP']
    df['Chol_Age'] = df['Cholesterol'] / (df['Age'] + 1)
    df['BP_MaxHR'] = df['BP'] / (df['Max HR'] + 1)
    df['ST_Slope'] = df['ST depression'] * df['Slope of ST']
    df['Vessels_Thallium'] = df['Number of vessels fluro'] * df['Thallium']
    df['Age_group'] = pd.cut(df['Age'], bins=[0,45,55,65,100], labels=[0,1,2,3]).astype(int)
    df['High_BP'] = (df['BP'] > 140).astype(int)
    df['High_Chol'] = (df['Cholesterol'] > 240).astype(int)
    df['Low_MaxHR'] = (df['Max HR'] < 140).astype(int)
    return df

X_train = add_features(X_train)
X_test = add_features(X_test)

# Handle missing values
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Scale features
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

print(f"\nFinal feature count: {X_train_scaled.shape[1]}")

#  10-Fold CV for better stability
n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Storage
oof_preds_xgb = np.zeros(len(X_train_scaled))
test_preds_xgb = np.zeros(len(X_test_scaled))
oof_preds_lgb = np.zeros(len(X_train_scaled))
test_preds_lgb = np.zeros(len(X_test_scaled))
oof_preds_cat = np.zeros(len(X_train_scaled))
test_preds_cat = np.zeros(len(X_test_scaled))
oof_preds_rf = np.zeros(len(X_train_scaled))
test_preds_rf = np.zeros(len(X_test_scaled))

cv_scores_xgb, cv_scores_lgb, cv_scores_cat, cv_scores_rf = [], [], [], []

print("\nStarting 10-Fold Cross-Validation Training...\n")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
    print(f"Fold {fold + 1}/{n_folds}")

    X_fold_train = X_train_scaled.iloc[train_idx]
    X_fold_val   = X_train_scaled.iloc[val_idx]
    y_fold_train = y_train.iloc[train_idx]
    y_fold_val   = y_train.iloc[val_idx]

    #  Tuned XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric='auc',
        early_stopping_rounds=50
    )
    xgb_model.fit(X_fold_train, y_fold_train,
                  eval_set=[(X_fold_val, y_fold_val)], verbose=False)
    oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_fold_val)[:, 1]
    test_preds_xgb += xgb_model.predict_proba(X_test_scaled)[:, 1] / n_folds
    cv_scores_xgb.append(roc_auc_score(y_fold_val, oof_preds_xgb[val_idx]))

    #  Tuned LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_fold_train, y_fold_train,
                  eval_set=[(X_fold_val, y_fold_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
    oof_preds_lgb[val_idx] = lgb_model.predict_proba(X_fold_val)[:, 1]
    test_preds_lgb += lgb_model.predict_proba(X_test_scaled)[:, 1] / n_folds
    cv_scores_lgb.append(roc_auc_score(y_fold_val, oof_preds_lgb[val_idx]))

    # Tuned CatBoost
    cat_model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.02,
        l2_leaf_reg=3,
        random_state=42,
        verbose=False,
        thread_count=-1,
        early_stopping_rounds=50,
        eval_metric='AUC'
    )
    cat_model.fit(X_fold_train, y_fold_train,
                  eval_set=(X_fold_val, y_fold_val), verbose=False)
    oof_preds_cat[val_idx] = cat_model.predict_proba(X_fold_val)[:, 1]
    test_preds_cat += cat_model.predict_proba(X_test_scaled)[:, 1] / n_folds
    cv_scores_cat.append(roc_auc_score(y_fold_val, oof_preds_cat[val_idx]))

    #  Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_fold_train, y_fold_train)
    oof_preds_rf[val_idx] = rf_model.predict_proba(X_fold_val)[:, 1]
    test_preds_rf += rf_model.predict_proba(X_test_scaled)[:, 1] / n_folds
    cv_scores_rf.append(roc_auc_score(y_fold_val, oof_preds_rf[val_idx]))

print("\n" + "="*50)
print("Cross-Validation Scores Summary:")
print("="*50)
print(f"XGBoost:       Mean AUC = {np.mean(cv_scores_xgb):.6f} (+/- {np.std(cv_scores_xgb):.6f})")
print(f"LightGBM:      Mean AUC = {np.mean(cv_scores_lgb):.6f} (+/- {np.std(cv_scores_lgb):.6f})")
print(f"CatBoost:      Mean AUC = {np.mean(cv_scores_cat):.6f} (+/- {np.std(cv_scores_cat):.6f})")
print(f"Random Forest: Mean AUC = {np.mean(cv_scores_rf):.6f} (+/- {np.std(cv_scores_rf):.6f})")

# Meta-model stacking
meta_train = pd.DataFrame({
    'xgb': oof_preds_xgb,
    'lgb': oof_preds_lgb,
    'cat': oof_preds_cat,
    'rf':  oof_preds_rf
})
meta_test = pd.DataFrame({
    'xgb': test_preds_xgb,
    'lgb': test_preds_lgb,
    'cat': test_preds_cat,
    'rf':  test_preds_rf
})

meta_model = LogisticRegression(max_iter=1000, random_state=42)
meta_model.fit(meta_train, y_train)

final_train_preds = meta_model.predict_proba(meta_train)[:, 1]
final_test_preds  = meta_model.predict_proba(meta_test)[:, 1]

print(f"\nMeta-model AUC Score: {roc_auc_score(y_train, final_train_preds):.6f}")

#  Rank average (often beats meta-model)
def rank_average(preds_list):
    ranks = np.zeros(len(preds_list[0]))
    for p in preds_list:
        ranks += pd.Series(p).rank(pct=True).values
    return ranks / len(preds_list)

rank_avg_test = rank_average([test_preds_xgb, test_preds_lgb, test_preds_cat, test_preds_rf])
rank_avg_oof  = rank_average([oof_preds_xgb,  oof_preds_lgb,  oof_preds_cat,  oof_preds_rf])
print(f"Rank Average OOF AUC: {roc_auc_score(y_train, rank_avg_oof):.6f}")

# Pick best between meta-model and rank average
meta_auc = roc_auc_score(y_train, final_train_preds)
rank_auc  = roc_auc_score(y_train, rank_avg_oof)

if rank_auc >= meta_auc:
    best_ensemble_test = rank_avg_test
    print(f"Using Rank Average (AUC: {rank_auc:.6f})")
else:
    best_ensemble_test = final_test_preds
    print(f"Using Meta-Model (AUC: {meta_auc:.6f})")

# Save submission
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
