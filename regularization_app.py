import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.datasets import make_regression

st.title("Regularization Demonstrator")

st.write("""
    This app demonstrates the concept of regularization in linear regression models. 
    Regularization helps to solve the overfitting problem by adding a penalty to the loss function. 
    We'll explore Ridge (L2) and Lasso (L1) regularization and their effects on model coefficients.
""")

# Sidebar settings for the app
st.sidebar.header("Regularization Settings")
st.sidebar.write("Adjust the lambda (λ) value to control the strength of regularization. \
    A higher λ value increases the regularization effect.")
lambda_value = st.sidebar.slider("Regularization strength (λ)", 0.0, 1.0, 0.1, 0.01)

st.sidebar.write("Set a threshold for the hard constraint (t). \
    Coefficients will be limited to not exceed this value in absolute terms.")
t_value = st.sidebar.slider("Hard constraint threshold (t)", 0.0, 10.0, 1.5, 0.1)

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
st.write("### Generated Synthetic Data for Demonstration")
st.write("The data used here is synthetically generated to illustrate the concepts of regularization.")

# Fit models without any regularization (OLS)
ols_coef = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
min_mse = ((y - X.dot(ols_coef)) ** 2).mean()

# Ridge Regression (L2 Regularization) with soft constraint
ridge_model = Ridge(alpha=lambda_value)
ridge_model.fit(X, y)
ridge_coef = ridge_model.coef_

# Lasso Regression (L1 Regularization) with soft constraint
lasso_model = Lasso(alpha=lambda_value)
lasso_model.fit(X, y)
lasso_coef = lasso_model.coef_

# Visualization of the loss function with and without regularization
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
st.write("### Visualization of Loss Functions")

# Loss function without regularization (OLS)
betas = np.linspace(ols_coef - 2, ols_coef + 2, 100)
loss_ols = [(np.sum((y - X.dot(beta)) ** 2) / len(y) - min_mse) for beta in betas]
ax[0].plot(betas, loss_ols, label='MSE Loss')
ax[0].axvline(x=t_value, color='red', linestyle='--', label='Hard Constraint (t)')
ax[0].set_title('Loss Function without Regularization')
ax[0].set_xlabel('Coefficient (Beta)')
ax[0].set_ylabel('Loss')
ax[0].legend()

# Loss function with L2 regularization
loss_ridge = [((np.sum((y - X.dot(beta)) ** 2) / len(y) - min_mse) + lambda_value * beta ** 2) for beta in betas]
ax[1].plot(betas, loss_ridge, label='MSE + L2 Loss')
ax[1].axvline(x=ridge_coef[0], color='green', linestyle='--', label='Ridge Coef')
ax[1].set_title('Loss Function with L2 Regularization')
ax[1].set_xlabel('Coefficient (Beta)')
ax[1].set_ylabel('Loss')
ax[1].legend()

st.pyplot(fig)

st.write("""
    ### Interpretation and Insights
    - **Hard vs. Soft Constraints:**  
      The hard constraint limits the absolute size of the coefficients, while the soft constraint 
      penalizes larger coefficients, influencing them towards smaller absolute values.
    - **Loss Function Visualization:**  
      The left plot shows the original loss function without regularization. 
      The right plot incorporates the regularization term, showing how it impacts the loss landscape.
    - **Effect of Regularization on Coefficients:**  
      As λ increases, the regularization effect becomes stronger, pulling the coefficients towards zero. 
      This helps in reducing overfitting but can increase underfitting if λ is too large.
    - **Current Coefficients with λ={lambda_value}:**  
      - Ridge Coefficient (soft constraint): {ridge_coef[0]:.2f}
      - Lasso Coefficient (soft constraint): {lasso_coef[0]:.2f}
    
    Adjust λ and t in the sidebar to see their effects on the model and loss function.
""")

# Adding an explanation guide for users
st.sidebar.write("""
    ### Guide
    - **λ (Lambda):** Controls the regularization strength. A value of 0 means no regularization.
    - **t (Threshold):** Used in visualizing the hard constraint in the loss function plots.
""")
