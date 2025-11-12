#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to run Lab 5: Bias-Variance Tradeoff
This script executes all the code from the notebook
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)

print("=" * 60)
print("Lab 5: Bias-Variance Tradeoff using Air Quality Dataset")
print("=" * 60)

# Step 1: Load and Prepare the Data
print("\nStep 1: Loading and preparing data...")
print("-" * 60)

# Load the dataset
df = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',')
print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# Select the features and target
features = ['T', 'RH', 'AH']
target = 'CO(GT)'

# Extract the selected columns
data = df[features + [target]].copy()

# Replace -200 with NaN (missing values)
data = data.replace(-200, np.nan)

# Convert columns to numeric (handles any remaining string issues)
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Remove rows with any missing values
data = data.dropna()

print(f"Data shape after removing missing values: {data.shape}")
print(f"\nSummary statistics:")
print(data.describe())

# Separate features and target
X = data[features].values
y = data[target].values

# Split into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")

# Step 2: Fit Models of Increasing Complexity
print("\n" + "=" * 60)
print("Step 2: Training polynomial regression models...")
print("-" * 60)

degrees = list(range(1, 11))  # Polynomial degrees from 1 to 10
train_errors = []
test_errors = []

for degree in degrees:
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    # Calculate Mean Squared Error
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    train_errors.append(train_mse)
    test_errors.append(test_mse)
    
    print(f"Degree {degree:2d}: Train MSE = {train_mse:.4f}, Test MSE = {test_mse:.4f}")

print("-" * 60)
print("Model training completed!")

# Step 3: Plot the Validation Curve
print("\n" + "=" * 60)
print("Step 3: Creating bias-variance tradeoff visualization...")
print("-" * 60)

# Find the optimal degree (minimum test error)
optimal_degree_idx = np.argmin(test_errors)
optimal_degree = degrees[optimal_degree_idx]
optimal_test_error = test_errors[optimal_degree_idx]

# Create the plot
plt.figure(figsize=(12, 8))
plt.plot(degrees, train_errors, 'o-', label='Training Error', linewidth=2, markersize=8)
plt.plot(degrees, test_errors, 's-', label='Testing Error', linewidth=2, markersize=8)

# Mark the optimal degree
plt.axvline(x=optimal_degree, color='red', linestyle='--', alpha=0.7, 
            label=f'Optimal Degree ({optimal_degree})')
plt.plot(optimal_degree, optimal_test_error, 'ro', markersize=12, 
         label=f'Minimum Test Error ({optimal_test_error:.4f})')

# Add annotations for regions
plt.axvspan(1, 3, alpha=0.1, color='blue', label='Underfitting Region')
plt.axvspan(optimal_degree-1, optimal_degree+1, alpha=0.1, color='green', 
           label='Optimal Complexity')
plt.axvspan(8, 10, alpha=0.1, color='red', label='Overfitting Region')

plt.xlabel('Model Complexity (Polynomial Degree)', fontsize=12, fontweight='bold')
plt.ylabel('Mean Squared Error (MSE)', fontsize=12, fontweight='bold')
plt.title('Bias–Variance Tradeoff: Training vs Testing Error', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(degrees)
plt.tight_layout()
plt.savefig('bias_variance_tradeoff.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'bias_variance_tradeoff.png'")
plt.close()

print(f"\nOptimal polynomial degree: {optimal_degree}")
print(f"Minimum test error: {optimal_test_error:.4f}")

# Optional: Cross-Validation Analysis
print("\n" + "=" * 60)
print("Bonus: Performing 5-fold cross-validation...")
print("-" * 60)

from sklearn.model_selection import cross_val_score

cv_train_errors = []
cv_test_errors = []

for degree in degrees:
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    
    # Train model
    model = LinearRegression()
    
    # Cross-validation (negative MSE, so we negate to get positive MSE)
    cv_scores = -cross_val_score(model, X_poly, y, cv=5, 
                                 scoring='neg_mean_squared_error')
    
    # Also compute on full training set for comparison
    model.fit(X_poly, y)
    train_pred = model.predict(X_poly)
    train_mse = mean_squared_error(y, train_pred)
    
    cv_train_errors.append(train_mse)
    cv_test_errors.append(cv_scores.mean())
    
    print(f"Degree {degree:2d}: CV MSE = {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

print("-" * 60)

# Plot cross-validation results
plt.figure(figsize=(12, 8))
plt.plot(degrees, cv_train_errors, 'o-', label='Training Error (Full Dataset)', 
         linewidth=2, markersize=8)
plt.plot(degrees, cv_test_errors, 's-', label='Cross-Validation Error (5-fold)', 
         linewidth=2, markersize=8)

# Find optimal degree from CV
optimal_cv_degree_idx = np.argmin(cv_test_errors)
optimal_cv_degree = degrees[optimal_cv_degree_idx]
optimal_cv_error = cv_test_errors[optimal_cv_degree_idx]

plt.axvline(x=optimal_cv_degree, color='red', linestyle='--', alpha=0.7,
            label=f'Optimal CV Degree ({optimal_cv_degree})')
plt.plot(optimal_cv_degree, optimal_cv_error, 'ro', markersize=12)

plt.xlabel('Model Complexity (Polynomial Degree)', fontsize=12, fontweight='bold')
plt.ylabel('Mean Squared Error (MSE)', fontsize=12, fontweight='bold')
plt.title('Bias–Variance Tradeoff: Cross-Validation Analysis', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(degrees)
plt.tight_layout()
plt.savefig('bias_variance_cv.png', dpi=300, bbox_inches='tight')
print("Cross-validation plot saved as 'bias_variance_cv.png'")
plt.close()

print(f"\nOptimal degree from cross-validation: {optimal_cv_degree}")
print(f"Optimal degree from train/test split: {optimal_degree}")
print(f"\nCross-validation provides a more robust estimate of model performance.")

print("\n" + "=" * 60)
print("Lab 5 completed successfully!")
print("=" * 60)

