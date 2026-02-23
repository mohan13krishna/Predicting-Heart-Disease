# Predicting Heart Disease - Ensemble Learning

An advanced machine learning solution for predicting heart disease using ensemble learning techniques. This project implements a meta-model stacking approach combining 5 different algorithms to achieve high ROC-AUC scores on the Kaggle Playground Series Season 6 Episode 2 competition. 

## 📊 Project Overview

**Competition**: [Kaggle Playground Series S6E2 - Predicting Heart Disease](https://kaggle.com/competitions/playground-series-s6e2)

**Goal**: Predict the likelihood of heart disease with high accuracy using ensemble learning

**Target Score**: ROC-AUC > 0.954

## 🏗️ Model Architecture

### Base Models (5)
- **XGBoost**: Gradient boosting with 500 estimators
- **LightGBM**: Fast gradient boosting framework
- **CatBoost**: Categorical boosting classifier
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sklearn gradient boosting

### Ensemble Strategy
- **Validation**: 5-Fold Stratified Cross-Validation
- **Meta-Model**: Logistic Regression stacking
- **OOF Predictions**: Out-of-fold predictions for training meta-model

## 📈 Expected Performance

| Method | AUC Score |
|--------|-----------|
| Simple Average | ~0.9535 |
| Weighted Average | ~0.9538 |
| Meta-Model (Stacking) | ~0.9542+ |

## 🚀 Quick Start

### Option 1: Run on Kaggle (Recommended)

1. Go to [Kaggle Notebooks](https://kaggle.com/code)
2. Create new notebook
3. Import from GitHub: `https://github.com/mohan13krishna/Predicting-Heart-Disease`
4. Add competition data as input: `playground-series-s6e2`
5. Enable GPU (optional but recommended)
6. Run all cells
7. Submit `submission.csv` to competition

### Option 2: Local Setup

```bash
# Clone repository
git clone https://github.com/mohan13krishna/Predicting-Heart-Disease.git
cd Predicting-Heart-Disease

# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm catboost

# Download datasets from Kaggle manually and place in data/ folder
# Run the script
python heart_disease_prediction.py
```

## 📁 Project Structure

```
Predicting-Heart-Disease/
├── heart_disease_ensemble.ipynb       # Jupyter notebook version
├── heart_disease_prediction.py        # Python script version (Kaggle-optimized)
├── README.md                          # This file
└── data/
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

## 🔧 Data Preprocessing

1. **Missing Values**: Handled with median imputation
2. **Feature Scaling**: RobustScaler normalization
3. **Target Encoding**: Categorical to numeric mapping
4. **Train-Test Split**: Stratified K-Fold cross-validation

## 📊 Feature Engineering

- All 30+ features from original dataset
- Robust scaling applied
- No feature interaction terms (can be added for improvement)

## 💾 Output

The model generates `submission.csv` with format:
```
id,Heart Disease
630000,0.9494
630001,0.0112
630002,0.9882
...
```

## ⏱️ Runtime

- **Kaggle GPU**: 10-15 minutes
- **Local CPU**: 30-60+ minutes (depends on hardware)

## 🎯 Key Features

✅ **Diverse Models**: 5 different algorithms capture different patterns  
✅ **Robust Validation**: 5-fold cross-validation prevents overfitting  
✅ **Meta-Model Stacking**: Logistic Regression learns optimal combination  
✅ **Kaggle Optimized**: Uses Kaggle competition data paths  
✅ **Production Ready**: Clean, documented, reproducible code  

## 🚀 How to Improve Score

1. **Hyperparameter Tuning**: Use GridSearchCV or Bayesian optimization
2. **Feature Engineering**: Create interaction terms, polynomial features
3. **Data Augmentation**: Handle class imbalance with upsampling
4. **More Ensemble Layers**: Add neural networks, SVM, etc.
5. **Probability Calibration**: Platt scaling, isotonic regression
6. **Multiple Seeds**: Try different random states and ensemble combinations
7. **LB Probing**: Track public/private LB to identify overfitting

## 📚 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0
```

## 📖 Notebook Details

### Cell 1: Data Loading
- Loads train.csv, test.csv, and sample_submission.csv
- Displays data shape, types, and missing values

### Cell 2: Exploratory Analysis
- Target distribution and class balance
- Statistical summary of features

### Cell 3: Data Preprocessing
- Feature extraction and scaling
- Missing value handling

### Cell 4: Target Encoding
- Converts categorical target to numeric (0, 1)

### Cell 5: Base Model Training
- 5-fold cross-validation
- Trains all 5 models simultaneously
- Tracks cross-validation scores

### Cell 6: Ensemble & Stacking
- Creates meta-features from base model predictions
- Trains meta-model (Logistic Regression)
- Compares different ensemble methods

### Cell 7: Submission Generation
- Creates final predictions
- Generates submission.csv in Kaggle format

## 🔗 Links

- [Kaggle Competition](https://kaggle.com/competitions/playground-series-s6e2)
- [GitHub Repository](https://github.com/mohan13krishna/Predicting-Heart-Disease)

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

Mohan Krishna

---

**Last Updated**: February 23, 2026  
**Status**: Ready for Kaggle Submission
