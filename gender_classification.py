from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
import numpy as np

#[height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

#['male', 'female']
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']
Y_true = ['male']

#For Neighbors Classifier
x = np.array(X)
y = np.array(Y)

#Set classifier
tree_clf = tree.DecisionTreeClassifier()
frst_clf = RandomForestClassifier(n_estimators=10)
ngbr_clf = NearestCentroid()

#Train them with data
tree_clf = tree_clf.fit(X, Y)
frst_clf = frst_clf.fit(X, Y)
ngbr_clf = ngbr_clf.fit(x, y)
NearestCentroid(metric='euclidean', shrink_threshold=None)

#Prediction
tree_predict = tree_clf.predict([[190, 82, 48]])
frst_predict = frst_clf.predict([[190, 82, 48]])
ngbr_predict = ngbr_clf.predict([[190, 82, 48]])

#Accuracy Score
tree_acc_score = accuracy_score(tree_predict, Y_true)
frst_acc_score = accuracy_score(frst_predict, Y_true)
ngbr_acc_score = accuracy_score(ngbr_predict, Y_true)

print(f'Tree Prediction: {tree_predict}, Accuracy Score: {tree_acc_score}')
print(f'Random Forest Prediction: {frst_predict}, Accuracy Score: {frst_acc_score}')
print(f'Nearest Neighbor Prediction: {ngbr_predict}, Accuracy Score: {ngbr_acc_score}')