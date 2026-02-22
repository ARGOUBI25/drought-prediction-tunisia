# compute_ssmi.py

import numpy as np
import pandas as pd

def compute_ssmi(data):
    """Compute the standardized soil moisture index (SSMI).

    Parameters:
    data (pd.Series): A pandas Series of soil moisture values.

    Returns:
    pd.Series: A pandas Series of SSMI values.
    """
    mean = data.mean()
    std_dev = data.std()
    ssmi = (data - mean) / std_dev
    return ssmi

if __name__ == '__main__':
    # Example usage
    soil_moisture_data = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    ssmi_values = compute_ssmi(soil_moisture_data)
    print(ssmi_values)