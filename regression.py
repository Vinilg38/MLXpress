from sklearn.tree import DecisionTreeRegressor
from MLXpress.diabetes import regression,predict,vis
model=DecisionTreeRegressor()
regression(model)
vis(model)
instance=[[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
predict(model,instance)
