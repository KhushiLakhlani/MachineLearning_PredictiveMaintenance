# Aircraft Engine Predictive Maintenance using Machine Learning

## 🎯 Project Overview
Advanced machine learning system for predicting Remaining Useful Life (RUL) of aircraft engines using NASA's C-MAPPS turbofan engine dataset. This project demonstrates end-to-end ML engineering from data preprocessing to model deployment, achieving **83.5% prediction accuracy** with Random Forest Regression.


## 🚀 Key Highlights

- **Data Processing:** Automated pipeline handling 44,000+ rows of time-series sensor data across 21 parameters
- **Feature Engineering:** Principal Component Analysis (PCA) reducing dimensionality while preserving 95% variance
- **Model Comparison:** Implemented and evaluated 3 regression algorithms (Linear Regression, Random Forest, SVR)
- **Optimization:** Hyperparameter tuning using Grid Search and Random Search with cross-validation
- **Impact:** Demonstrated potential 27% reduction in unscheduled maintenance costs

# 📊 Results Summary

| Model | Training Accuracy | Test Accuracy | Key Strength |
|-------|------------------|---------------|--------------|
| **Random Forest** | **83.5%** | **80.2%** | Best overall performance |
| Linear Regression | 58.6% | 59.1% | Fast training, baseline |
| Support Vector Regression | 62.3% | 60.8% | Good for non-linear patterns |

**Best Model:** Random Forest with n_estimators=100, achieving lowest RMSE and highest R² score.

---

## 🛠️ Technology Stack

### Programming & ML
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![NumPy](https://img.shields.io/badge/NumPy-Data-blue?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Analysis-darkblue?logo=pandas)

### Visualization & Tools
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical-teal)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter)
