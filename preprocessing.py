import numpy as np
from MLXpress.preprocessing import handle_missing_values,perform_cross_validation,select_best_features,scale_data,remove_low_variance_features,split_data
import scipy.stats

import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset as an example
iris = load_iris()
X = iris.data
y = iris.target

# Convert the data into pandas DataFrame
X = pd.DataFrame(X, columns=iris.feature_names)
y = pd.Series(y, name='target')
# Handle missing values
handle_missing_values(X)

# Perform cross-validation
perform_cross_validation(X, y)

# Select best features
select_best_features(X, y,2)

# Scale data
scale_data(X)

# Remove low variance features
remove_low_variance_features(X)

# Split data
split_data(X, y)
