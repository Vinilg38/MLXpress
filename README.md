# MLXpress Test Files
This repository serves the purpose of demonstrating the usage of MLXpress Library.
It consists of the following subpackages:

**iris.py:**
The iris.py subpackage is used for performing classification and clustering tasks. It provides functions for visualizing the data, training classification models, and evaluating their performance.

**wine.py:**
The wine.py subpackage is used for classification tasks. It offers functions for training classification models, visualizing the data, and assessing model accuracy.

**diabetes.py:**
The diabetes.py subpackage is used for regression analysis. It includes functions for regression model training, prediction, and evaluation.

**preprocessing.py:**
The preprocessing.py subpackage provides common preprocessing tasks. It includes functions for handling missing values, scaling data, feature selection, and splitting datasets into training and testing sets.

**stats.py:**
The stats.py subpackage contains statistical analysis functions. It includes functions for calculating basic statistics, analyzing data distribution, and visualizing data using box plots and histograms.

**math.py:**
The math.py subpackage provides mathematical functions commonly used in machine learning. It includes functions for calculating distances, performing hypothesis tests, and working with covariance matrices.

To use any of the sub packages, import the desired functions into your code. 

``` python
from MLXpress.iris import classification, vis
from sklearn.tree import DecisionTreeClassifier

# Create a classification model
model = DecisionTreeClassifier()

# Perform classification
classification(model)

# Visualize the data
vis(model)
```


Do not refrain from using the inbuilt help() function,whenever stuck:
```python
help(scale_data)```


