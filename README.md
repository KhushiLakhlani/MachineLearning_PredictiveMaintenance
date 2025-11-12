# Aircraft Engine Predictive Maintenance using Machine Learning

<img src="images/model_comparison.png" alt="Model Performance Comparison" width="100%">

## 🎯 Project Overview
Advanced machine learning system for predicting Remaining Useful Life (RUL) of aircraft engines using NASA's C-MAPPS turbofan engine dataset. This project demonstrates end-to-end ML engineering from data preprocessing to model deployment, achieving **83.5% prediction accuracy** with Random Forest Regression.


## 🚀 Key Highlights

- **Data Processing:** Automated pipeline handling 44,000+ rows of time-series sensor data across 21 parameters
- **Feature Engineering:** Principal Component Analysis (PCA) reducing dimensionality while preserving 95% variance
- **Model Comparison:** Implemented and evaluated 3 regression algorithms (Linear Regression, Random Forest, SVR)
- **Optimization:** Hyperparameter tuning using Grid Search and Random Search with cross-validation
- **Impact:** Demonstrated potential 27% reduction in unscheduled maintenance costs

## 📊 Results Summary

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


## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook

### Quick Start
```bash
# Clone the repository
git clone https://github.com/KhushiLakhlani/MachineLearning_PredictiveMaintenance.git
cd MachineLearning_PredictiveMaintenance

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

### Run the Analysis
1. Open `notebooks/01_Data_Exploration.ipynb`
2. Run all cells sequentially
3. Continue with notebooks 02 and 03

---

## 📈 Key Features

### 1. Advanced Data Preprocessing
- Handled missing values and outlier detection
- Normalized sensor readings across 21 measurements
- Created derived metrics for engine degradation patterns

### 2. Feature Engineering
- **PCA:** Reduced 21 sensor dimensions while preserving 95% variance
- **Temporal Features:** Extracted cycle-based degradation indicators
- **Custom Metrics:** Calculated remaining useful life (RUL) targets

### 3. Model Development
- **Linear Regression:** Baseline model for performance comparison
- **Random Forest:** Ensemble learning with 100 estimators
- **Support Vector Regression:** Non-linear kernel-based approach

### 4. Hyperparameter Optimization
- Grid Search with 3-fold cross-validation
- Random Search for efficient parameter exploration
- Model evaluation using MAE, MSE, RMSE, and R² metrics

---

## 📊 Dataset Information

**Source:** NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)

**Subset Used:** FD003
- 100 training engine trajectories
- 100 testing engine trajectories
- 21 sensor measurements per cycle
- 3 operational settings
- 2 fault modes: High-Pressure Compressor (HPC) and Fan degradation

**Key Sensors Analyzed:**
- Temperature sensors (T2, T24, T30, T50)
- Pressure sensors (P2, P15, P30, Ps30)
- Physical fan speeds (Nf, Nc)
- Bleed enthalpy, fuel flow, and more

---
## 💡 Key Insights

### Technical Findings
- Random Forest outperformed linear methods due to data's non-linear nature
- PCA effectively reduced computational complexity without sacrificing accuracy
- Early failure detection possible 50+ cycles before actual engine failure

### Business Impact
- Predictive maintenance reduces unscheduled downtime by ~70%
- Potential cost savings of 27% compared to preventive maintenance
- Improved flight safety through proactive component replacement


## 🔮 Future Enhancements

- [ ] Implement LSTM neural networks for temporal sequence modeling
- [ ] Develop real-time prediction API for production deployment
- [ ] Extend to multi-fault classification scenarios
- [ ] Create interactive dashboard using Streamlit/Plotly
- [ ] GPU-based inference for faster predictions
