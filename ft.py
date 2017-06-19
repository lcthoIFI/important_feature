

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor
import numpy as np
#Load boston housing dataset as an example
boston = load_iris()
X = boston["data"]
print(X)
Y = boston["target"]
print(Y)
names = boston["feature_names"]
#print(names)
rf = RandomForestRegressor()
rf.fit(X, Y)
print ("Features sorted by their score:")
print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True))