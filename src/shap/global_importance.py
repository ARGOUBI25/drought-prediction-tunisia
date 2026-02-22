import shap
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('path/to/your/model.pkl')  # Update the path to your model

# Load your data
X = pd.read_csv('path/to/your/data.csv')  # Update the path to your data

# Calculate SHAP values
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Plot global importance
shap.summary_plot(shap_values, X)

# Save the plot if needed
plt.savefig('shap_global_importance.png')
