from MLXpress.iris import classification,vis,predict
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
classification(model)
instance= [[0.1,0.8,0.9,0.9]]
predict(model,instance)
vis(model)