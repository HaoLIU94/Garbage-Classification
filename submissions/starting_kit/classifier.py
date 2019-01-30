from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier



class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = RandomForestClassifier(
            n_estimators=10, max_leaf_nodes=10, random_state=42)

    def fit(self, X, y):
    	X1 = X.reshape((X.shape[0],-1))
    	self.clf.fit(X1, y)

    def predict_proba(self, X):
    	X1 = X.reshape((X.shape[0],-1))
    	y_pred_proba = self.clf.predict_proba(X1)
    	return y_pred_proba
