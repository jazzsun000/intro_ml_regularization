import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

st.title("Regularization and Coefficient Shrinkage Demonstrator")

st.write("""
    This application visualizes the effect of regularization on a regression model. 
    Regularization is a technique used to prevent overfitting by penalizing large coefficients. 
    In this demonstration, we use Ridge regression, which applies L2 regularization.
""")

# Sidebar settings for the app
st.sidebar.header("Regularization Settings")
st.sidebar.write("""
    Adjust the sliders to change the maximum value of lambda and the number of lambda values to test. 
    Lambda controls the strength of the regularization.
""")
max_lambda = st.sidebar.slider("Max Lambda Value", 0.0, 10.0, 2.0, 0.1)
num_lambdas = st.sidebar.slider("Number of Lambda Values", 2, 100, 20)

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Explain the data generation process
st.write("""
    ### Data Generation
    For demonstration purposes, we generate synthetic regression data. 
    This dataset is then split into training and testing sets.
""")

# Prepare to record coefficient sizes and errors
lambdas = np.logspace(-4, np.log10(max_lambda), num_lambdas)
coefs = []
train_errors = []
test_errors = []

# Train models and record metrics
for l in lambdas:
    model = Ridge(alpha=l)
    model.fit(X_train, y_train)
    coefs.append(model.coef_[0])
    train_errors.append(np.mean((model.predict(X_train) - y_train) ** 2))
    test_errors.append(np.mean((model.predict(X_test) - y_test) ** 2))

# Plot coefficient shrinkage
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].plot(lambdas, coefs)
ax[0].set_xscale('log')
ax[0].set_xlabel('Lambda (Regularization strength)')
ax[0].set_ylabel('Coefficient Value')
ax[0].set_title('Coefficient Shrinkage as Lambda Increases')

# Plot training and validation error
ax[1].plot(lambdas, train_errors, label='Training Error')
ax[1].plot(lambdas, test_errors, label='Test Error')
ax[1].set_xscale('log')
ax[1].set_xlabel('Lambda (Regularization strength)')
ax[1].set_ylabel('Mean Squared Error')
ax[1].set_title('Training and Test Error as Lambda Increases')
ax[1].legend()

st.pyplot(fig)

st.write("""
    ### Analysis and Interpretation
    - **Coefficient Shrinkage:** As the regularization strength (lambda) increases, 
      the coefficients (betas) shrink towards zero. This effect is more pronounced as lambda 
      becomes larger, demonstrating the trade-off between model complexity and regularization.
    - **Training and Validation Error:** With increasing lambda, the training error generally 
      increases due to the model becoming less flexible. The validation (or test) error may decrease 
      initially if overfitting is mitigated but will start to increase again if the model becomes underfit 
      with too much regularization.
    - **Left Plot:** Shows how the beta coefficient of the model shrinks as lambda increases.
    - **Right Plot:** Displays the corresponding training and test error, providing insights into 
      the model's performance and the potential for overfitting or underfitting at different levels of regularization.
""")

st.write("""
    Use the sliders in the sidebar to experiment with different regularization strengths and observe 
    how they impact the model's performance and coefficient values.
""")
