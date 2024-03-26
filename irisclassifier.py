from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRgression
from sklearn.model_selection import train_test_split
import pickle as pickle

iris = load_iris()
X, y = iris.data, iris.target


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
print(X_train.shape)
print(y_train.shape)

clf = RandomForestClassifier()
# clf = LogisticRegression()
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

print("Scroring model to pickle file")
pickle.dump(clf,open("iris_model.pkl", 'wb'))
