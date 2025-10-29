# Predictive Analytics - C-MAPPS Dataset
# Save this as a .py file or convert to .ipynb

# ============================================
# IMPORTING LIBRARIES
# ============================================

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point

#Importing Common Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer

import numpy as np
from sklearn import metrics
from numpy import mean
from numpy import std
import time
import warnings

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import make_regression

# ============================================
# IMPORTING C-MAPPS DATASET
# ============================================

#Importing Data
df = pd.read_csv(r"C:\Users\Khushi Lakhlani\Desktop\Dissertation\G001CEM\CMAPPS_Data\train_FD003.txt", sep=' ', header=None)

# ============================================
# DATA PREPARATION
# ============================================

#Training data
df.columns = ['unit_number', 'cycle', 'ops_1', 'ops_2', 'ops_3', 'sm_1',
             'sm_2', 'sm_3', 'sm_4', 'sm_5', 'sm_6', 'sm_7', 'sm_8', 'sm_9', 'sm_10', 'sm_11', 'sm_12', 'sm_13', 'sm_14',
             'sm_15', 'sm_16', 'sm_17', 'sm_18', 'sm_19', 'sm_20', 'sm_21', 'sm_22', 'sm_23']

#Dropping columns containing NaN values
df.drop('sm_22', axis=1, inplace=True)
df.drop('sm_23', axis=1, inplace=True)

df.head()

rul_df = pd.read_csv(r"C:\Users\Khushi Lakhlani\Desktop\Dissertation\G001CEM\CMAPPS_Data\RUL_FD003.txt")
rul_df.columns = ['time-to-failure']
rul_df.head()

df.isnull().sum()

df.head()

#drop any null values in the dataset
df.dropna(inplace=True)
#reset the index after dropping null values
df.reset_index(drop=True, inplace=True)

# Define a filter function
def ignore_warning(message, category, filename, lineno, file=None, line=None):
    return 'Precision loss occurred in moment calculation due to catastrophic cancellation' in str(message)

# Suppress the warning
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)

# Apply the filter function
warnings.filterwarnings("ignore", message=".*catastrophic cancellation.*", category=RuntimeWarning)
from scipy.stats import skew
skewness_values = df.apply(lambda x: skew(x))
skewness_values

#Feature Selection
features = ['ops_1', 'ops_2', 'sm_2', 'sm_3', 'sm_4', 'sm_6', 'sm_7', 'sm_8', 'sm_9', 'sm_11',
           'sm_12', 'sm_13', 'sm_14', 'sm_15', 'sm_17', 'sm_20']

#Histograms
# Define the number of columns for the grid layout
num_cols = 16 #number of features

# Calculate the number of rows needed based on the number of features and number of columns
num_features = len(features)
num_rows = (num_features + num_cols - 1) // num_cols

# Plot histograms for each feature
for i, feature in enumerate(features):
    # Create a new figure for each feature
    fig = plt.figure(figsize=(8, 6))
    
    # Plot histogram for the current feature
    df[feature].hist(bins=20)
    
    # Set the title for the subplot
    plt.title(feature)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()

# ============================================
# FEATURE EXTRACTION
# ============================================

rul_df = pd.DataFrame(df.groupby('unit_number')['cycle'].max()).reset_index()
rul_df.columns = ['unit_number', 'max_cycle']  # Rename columns to avoid conflict

# Drop duplicate columns from df_train, if any
df = df.drop(columns=['max_cycle'], errors='ignore')

# Merge dataframes
df = df.merge(rul_df, on=['unit_number'], how='left')

# Calculate RUL
df['RUL'] = df['max_cycle'] - df['cycle']

# Drop unnecessary column
df = df.drop('max_cycle', axis=1)

# Display the first few rows of the dataframe
df.head()

#Feature Selection
features = ['ops_1', 'ops_2', 'sm_2', 'sm_3', 'sm_4', 'sm_6', 'sm_7', 'sm_8', 'sm_9', 'sm_11',
           'sm_12', 'sm_13', 'sm_14', 'sm_15', 'sm_17', 'sm_20', 'sm_21']

# ============================================
# DATA SPLITTING
# ============================================

#Setting X as training data (feature)
X = df[features]

#Setting y as testing data (target)
y = df['RUL']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Shape of Train dataset
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

#Number of rows and columns in test set
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

# ============================================
# DATA VISUALISATION
# ============================================

sns.set(rc={'figure.figsize':(3, 3)})  # Adjust figure size

# Select a subset of unique ops_1 values or aggregate them
selected_ops_1 = df['ops_1'].value_counts().nlargest(5).index
df_filtered = df[df['ops_1'].isin(selected_ops_1)]

# Create countplot with limited ops_1 values
sns.countplot(x="sm_1", hue="ops_1", data=df_filtered, color="darkorange" )
plt.title("Sensor 1 performance in one operational cycle")
plt.show()

sensor_columns = ["sm_1", "sm_2", "sm_3", "sm_4", "sm_5", "sm_6", "sm_7",
                 "sm_8", "sm_9", "sm_10", "sm_11", "sm_12", "sm_13", "sm_14",
                 "sm_15", "sm_16", "sm_17", "sm_18", "sm_19", "sm_20",
                 "sm_21"]

# Set up subplots
fig, axes = plt.subplots(nrows=len(sensor_columns), figsize=(12, 3 * len(sensor_columns)))

for ax, column in zip(axes, sensor_columns):
    df[column] = pd.to_numeric(df[column], errors='coerce')
    sns.histplot(data=df, x=column, kde=True, label=column, bins=30, ax=ax)
    ax.set_xlabel("Sensor Data Values")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram of {column}")

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()

corr_matrix = df.corr()
plt.figure(figsize=(24,24))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Extracting latitude and longitude coordinates from the selected features
latitude = X['ops_1']
longitude = X['ops_2']

# Create a hexbin plot to visualize sensor density
fig, ax = plt.subplots(figsize=(10, 7))
hb = ax.hexbin(longitude, latitude, gridsize=50, cmap='Blues', mincnt=1)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Sensor Network Map (Hexbin)')
cb = fig.colorbar(hb, ax=ax)
cb.set_label('Density')
plt.show()

# PCA
from sklearn.decomposition import PCA

pca_model = PCA(n_components=2)
X_pca = pca_model.fit_transform(X)

# Dimensionality Reduction Visualization
fig, ax = plt.subplots(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=10)
plt.title('PCA Visualization of Features')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

import mpl_toolkits.mplot3d.axes3d as axes3d

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X['sm_2'], X['sm_3'], X['sm_4'], c=y, cmap='viridis')

plt.tight_layout()
plt.show()

import networkx as nx
import itertools

def create_network_graph(df, threshold=0.5):
    G = nx.Graph()
    G.add_nodes_from(df.columns)
    edges = []
    for col1, col2 in itertools.combinations(df.columns, 2):
        if df[col1].corr(df[col2]) > threshold:
            edges.append((col1, col2))
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, k=1.5)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    plt.axis('off')
    plt.show()

create_network_graph(X.corr())

#Importing Linear Regression Library
from sklearn.linear_model import LinearRegression

# Define the features and target data
features = ['ops_1', 'ops_2', 'sm_2', 'sm_3', 'sm_4', 'sm_6', 'sm_7', 'sm_8', 'sm_9', 'sm_11',
           'sm_12', 'sm_13', 'sm_14', 'sm_15', 'sm_17', 'sm_20', 'sm_21']
target = [44, 51, 27, 120, 101, 99, 71, 55, 55, 66, 77, 115, 115, 31, 108, 56, 136, 132, 85, 56, 18,
         119, 78, 9, 58, 11, 88, 144, 124, 89, 79, 55, 71, 65, 87, 137, 145, 22, 8, 41, 131, 115,
         128, 69, 111, 7, 137, 55, 135, 11, 78, 120, 87, 87, 55, 93, 88, 40, 49, 128, 129, 58,
         117, 28, 115, 87, 92, 103, 100, 63, 35, 45, 99, 117, 45, 27, 86, 20, 18, 133, 15, 6, 145,
         104, 56, 25, 68, 144, 41, 51, 81, 14, 67, 18, 127, 113, 123, 17, 8, 28]

# Convert the target list to a numpy array
target = np.array(target)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Get the coefficients of the model
coefficients = model.coef_

# Calculate the contribution of each feature to the overall engine performance
feature_contributions = np.abs(coefficients) / np.sum(np.abs(coefficients))

# Sort the feature contributions in descending order
sorted_indices = np.argsort(feature_contributions)[::-1]
sorted_contributions = feature_contributions[sorted_indices]
sorted_features = np.array(features)[sorted_indices]

# Calculate the cumulative contributions
cumulative_contributions = np.cumsum(sorted_contributions)

# Plot the Pareto chart
plt.figure(figsize=(10, 6))
plt.bar(range(len(features)), sorted_contributions, align='center', alpha=0.7)
plt.plot(range(len(features)), cumulative_contributions, color='r', marker='o')
plt.xticks(range(len(features)), sorted_features, rotation=90)
plt.xlabel('Features')
plt.ylabel('Contribution')
plt.title('Pareto Chart of Feature Contributions')
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================
# LINEAR REGRESSION
# ============================================

#Importing Linear Regression Library
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

#Make predictions on test set
y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

abs_errors = np.abs(y_test - y_test_pred)

plt.hist(abs_errors, bins=50)
plt.xlabel('Absolute Error')
plt.ylabel('Frequency')
plt.title('Histogram of Absolute Errors')
plt.show()

#training score - linear regression
train_score = lr_model.score(X_train, y_train)
print("Training Score: {:.3f}".format(train_score))
#testing score - linear regression
test_score = lr_model.score(X_test, y_test)
print("Test Score:{:.3f}".format(test_score))

# ============================================
# DATA SCALING
# ============================================

#Standard Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Linear Regression performed after scaling

lr_model.fit(X_train_scaled,y_train)

y_train_pred = lr_model.predict(X_train_scaled)
y_test_pred = lr_model.predict(X_test_scaled)

#Scaled training score
lr_model_train_scaled_score = lr_model.score(X_train_scaled, y_train)
print("Scaled Training Score: {:.3f}".format(lr_model_train_scaled_score))

#Scaled testing score
lr_model_test_scaled_score = lr_model.score(X_test_scaled, y_test)
print("Scaled Test Score: {:.3f}".format(lr_model_test_scaled_score))

# ============================================
# L2 REGULARIZATION - RIDGE REGRESSION
# ============================================

from sklearn.linear_model import Ridge

#Initilize Ridge regression model
ridge_model = Ridge(alpha=1.0)

#Fit the model to training data
ridge_model.fit(X_train_scaled, y_train)

#Make predictions on training set
y_train_pred_ridge = ridge_model.predict(X_train_scaled)

#Calculate MSE on training set
mse_train_ridge = mean_squared_error(y_train, y_train_pred_ridge)
print("Train MSE (Ridge): ", mse_train_ridge)

# ============================================
# GRID SEARCH - LINEAR REGRESSION
# ============================================

# Define the parameter grid for Grid Search
param_grid = {
    'fit_intercept': [True, False],
    'n_jobs': [None, -1, 1, 2],
    'positive': [True, False]
}

start = time.time()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=LinearRegression(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Perform grid search
grid_search.fit(X_train, y_train)

end = time.time()

# Best parameters found during grid search
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Get the best estimator
best_estimator = grid_search.best_estimator_

# Training score with the best estimator
best_estimator_train_score = best_estimator.score(X_train, y_train)
print("Training score with best estimator: {:.3f}".format(best_estimator_train_score))

# Test score with the best estimator
best_estimator_test_score = best_estimator.score(X_test, y_test)
print("Test score with best estimator: {:.3f}".format(best_estimator_test_score))

print("Time taken to evaluate and find best hyperparameters: {:.2f} seconds".format(end - start))

# ============================================
# RANDOM SEARCH - LINEAR REGRESSION
# ============================================

# Define the parameter distributions to search
lr_param_dist = {
    'fit_intercept': [True, False],
    'n_jobs': [None, -1, 1, 2],
    'positive': [True, False]
}

start = time.time()

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=LinearRegression(), param_distributions=lr_param_dist, n_iter=100,
                                  cv=5, n_jobs=-1, verbose=2, random_state=42)

# Perform random search
random_search.fit(X_train, y_train)

end = time.time()

# Best parameters found during random search
best_params = random_search.best_params_
print("Best Parameters:", best_params)

# Get the best estimator
best_estimator = random_search.best_estimator_

# Training Score with best estimator
best_estimator_train_score = best_estimator.score(X_train, y_train)
print("Training score with best estimator: {:.3f}".format(best_estimator_train_score))

# Test Score with best estimator
best_estimator_test_score = best_estimator.score(X_test, y_test)
print("Test score with best estimator: {:.3f}".format(best_estimator_test_score))

print("Time taken to evaluate and find best hyperparameters: {:.2f} seconds".format(end - start))

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_test_pred)

# Print the evaluation metrics
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-Squared (R2): ", r2)

# ============================================
# RESIDUAL PLOT
# ============================================

# Calculate residuals
residuals = y_train - y_train_pred

# Plot residual plot
plt.figure(figsize=(10, 6))
plt.scatter(y_train_pred, residuals, alpha=0.5)
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')  # Add horizontal line at y=0
plt.grid(True)
plt.show()

# Plotting actual vs predicted for training set
plt.figure(figsize=(10, 5))
plt.scatter(y_train, y_train_pred, color='blue', label='Actual vs Predicted (Training)')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted (Training)')
plt.legend()
plt.grid(True)
plt.show()

# Plotting actual vs predicted for test set
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_test_pred, color='green', label='Actual vs Predicted (Test)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted (Test)')
plt.legend()
plt.grid(True)
plt.show()

# ============================================
# RANDOM FOREST REGRESSION
# ============================================

#Importing Random Forest library
from sklearn.ensemble import RandomForestRegressor

#Initialising the regressor
rf_model = RandomForestRegressor()

# Training the model
rf_model.fit(X_train, y_train)

#Training Score
rf_model_train_score = rf_model.score(X_train, y_train)
print("Training score: {:.3f}".format(rf_model_train_score))

#Test Score
rf_model_test_score = rf_model.score(X_test, y_test)
print("Test score: {:.3f}".format(rf_model_test_score))

# ============================================
# GRID SEARCH - RANDOM FOREST
# ============================================

# Define the hyperparameters grid
grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

start = time.time()

#initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=grid_rf, cv=3, n_jobs=-1, verbose=2)

#Perform Grid Search
grid_search.fit(X_train, y_train)

end = time.time()

# Best parameters found during grid search
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Get the best estimator
best_estimator = grid_search.best_estimator_

# Training Score with best estimator
best_estimator_train_score = best_estimator.score(X_train, y_train)
print("Training score with best estimator: {:.3f}".format(best_estimator_train_score))

# Test Score with best estimator
best_estimator_test_score = best_estimator.score(X_test, y_test)
print("Test score with best estimator: {:.3f}".format(best_estimator_test_score))

print("Time taken to evaluate and find best hyperparameters: {:.2f} seconds".format(end - start))

# ============================================
# RANDOM SEARCH - RANDOM FOREST
# ============================================

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define the parameter distributions to search
random_rf = {
    'n_estimators': randint(50, 300),
    'max_depth': [None] + list(randint(2, 10).rvs(3)),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10)
}

start = time.time()

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=random_rf, n_iter=100,
                                  cv=3, n_jobs=-1, verbose=2, random_state=42)

# Perform random search
random_search.fit(X_train, y_train)

end = time.time()

# Best parameters found during random search
best_params = random_search.best_params_
print("Best Parameters:", best_params)

# Get the best estimator
best_estimator = random_search.best_estimator_

# Training Score with best estimator
best_estimator_train_score = best_estimator.score(X_train, y_train)
print("Training score with best estimator: {:.3f}".format(best_estimator_train_score))

# Test Score with best estimator
best_estimator_test_score = best_estimator.score(X_test, y_test)
print("Test score with best estimator: {:.3f}".format(best_estimator_test_score))

print("Time taken to evaluate and find best hyperparameters: {:.2f} seconds".format(end - start))

# ============================================
# CROSS-VALIDATION - RANDOM FOREST
# ============================================

# Perform cross-validation to evaluate the model's performance
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=3)
print("Cross-Validation Scores:", cv_scores)
print("Average CV Score:", np.mean(cv_scores))

# ============================================
# SUPPORT VECTOR REGRESSION (SVR)
# ============================================

from sklearn.svm import SVR

# SVR model
svr_model = SVR()

# Fit the SVR model
svr_model.fit(X_train, y_train)

#training score - logistic regression
svr_model_train_score = svr_model.score(X_train, y_train)
print("Training Score: {:.3f}".format(train_score))
#testing score - logistic regression
svr_model_test_score = svr_model.score(X_test, y_test)
print("Test Score:{:.3f}".format(test_score))

y_train_pred_svr = svr_model.predict(X_train)
y_test_pred_svr = svr_model.predict(X_test)

# ============================================
# CROSS-VALIDATION - SVR
# ============================================

# Perform cross-validation on the scaled training data
start = time.time()
cv_scores = cross_val_score(svr_model, X_train_scaled, y_train, cv=5)

end = time.time()
# Print the cross-validation results
print("Cross-validation results for SVR model with non-linear kernel:", cv_scores)
print("Average cross-validation score: {:.3f}".format(cv_scores.mean()))
print("Time taken to perform cross-validation: ", (end-start))

# ============================================
# GRID SEARCH - SVR
# ============================================

# Define the hyperparameters grid
grid_svr = {
    'C': [0.1],
    'kernel': ['linear'],
    'gamma': ['scale'],
    'epsilon': [0.1]
}

start = time.time()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=svr_model, param_grid=grid_svr, cv=3, n_jobs=-1, verbose=2)

# Perform Grid Search
grid_search.fit(X_train, y_train)

end = time.time()

# Best parameters found during grid search
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Get the best estimator
best_estimator = grid_search.best_estimator_

# Training Score with best estimator
best_estimator_train_score = best_estimator.score(X_train, y_train)
print("Training score with best estimator: {:.3f}".format(best_estimator_train_score))

# Test Score with best estimator
best_estimator_test_score = best_estimator.score(X_test, y_test)
print("Test score with best estimator: {:.3f}".format(best_estimator_test_score))

print("Time taken to evaluate and find best hyperparameters: {:.2f} seconds".format(end - start))

# ============================================
# RANDOM SEARCH - SVR
# ============================================

# Define the hyperparameters grid
random_svr = {
    'C': [0.1],
    'kernel': ['linear'],
    'gamma': ['scale'],
    'epsilon': [0.1]
}

start = time.time()

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=svr_model, param_distributions=random_svr, n_iter=10, cv=3, n_jobs=-1, verbose=2,
                                  random_state=42)

# Perform Randomized Search
random_search.fit(X_train, y_train)

end = time.time()

# Best parameters found during random search
best_params = random_search.best_params_
print("Best Parameters:", best_params)

# Get the best estimator
best_estimator = random_search.best_estimator_

# Training Score with best estimator
best_estimator_train_score = best_estimator.score(X_train, y_train)
print("Training score with best estimator: {:.3f}".format(best_estimator_train_score))

# Test Score with best estimator
best_estimator_test_score = best_estimator.score(X_test, y_test)
print("Test score with best estimator: {:.3f}".format(best_estimator_test_score))

print("Time taken to evaluate and find best hyperparameters: {:.2f} seconds".format(end - start))

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_test_pred_svr)
mse = mean_squared_error(y_test, y_test_pred_svr)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_test_pred_svr)

# Print the evaluation metrics
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-Squared (R2): ", r2)

# Plotting actual vs predicted for training set
plt.figure(figsize=(10, 5))
plt.scatter(y_train, y_train_pred_svr, color='blue', label='Actual vs Predicted (Training)')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted (Training) - Support Vector Regression')
plt.legend()
plt.grid(True)
plt.show()

# Plotting actual vs predicted for test set
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_test_pred_svr, color='green', label='Actual vs Predicted (Test)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted (Test) - Support Vector Regression')
plt.legend()
plt.grid(True)
plt.show()