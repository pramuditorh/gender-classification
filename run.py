from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors.nearest_centroid import NearestCentroid
import numpy as np

#[height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

#['male', 'female']
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

#For Neighbors Classifier
x = np.array(X)
y = np.array(Y)

#Set classifier
tree_clf = tree.DecisionTreeClassifier()
frst_clf = RandomForestClassifier(n_estimators=10)
ngbr_clf = NearestCentroid()
svm_clf = SVC(gamma='scale')

#Train them with data
tree_clf = tree_clf.fit(X, Y)
frst_clf = frst_clf.fit(X, Y)
ngbr_clf = ngbr_clf.fit(x, y)
NearestCentroid(metric='euclidean', shrink_threshold=None)
svm_clf = svm_clf.fit(X, Y)

#Prediction
tree_predict = tree_clf.predict(X)
frst_predict = frst_clf.predict(X)
ngbr_predict = ngbr_clf.predict(x)
svm_predict = svm_clf.predict(X)

#Accuracy Score
tree_acc_score = accuracy_score(Y, tree_predict)
frst_acc_score = accuracy_score(Y , frst_predict)
ngbr_acc_score = accuracy_score(Y, ngbr_predict)
svm_acc_score = accuracy_score(Y, svm_predict)

print(f'Tree Prediction: {tree_predict}, Accuracy Score: {tree_acc_score}')
print(f'Random Forest Prediction: {frst_predict}, Accuracy Score: {frst_acc_score}')
print(f'Nearest Neighbor Prediction: {ngbr_predict}, Accuracy Score: {ngbr_acc_score}')
print(f'SVM Prediction: {svm_predict}, Accuracy Score: {svm_acc_score}')