from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier


class Classifier(BaseEstimator):
    def __init__(self):
        # self.clf = RandomForestClassifier(
        #     n_estimators=10, max_leaf_nodes=10, random_state=42)
        self.clf = xgb.XGBClassifier(learning_rate = 0.1, max_depth = 1000, alpha = 10, n_estimators = 1000, max_leaf_nodes=1000) 
        # self.clf = LGBMClassifier(boosting_type='gbdt', n_estimators=70, num_leaves=8, max_depth=7, learning_rate=0.1, subsample=0.6, colsample_bytree=0.6, objective='binary')

    def fit(self, X, y):
    	X1 = X.reshape((X.shape[0],-1))
    	self.clf.fit(X1, y)

    def predict_proba(self, X):
    	X1 = X.reshape((X.shape[0],-1))
    	y_pred_proba = self.clf.predict_proba(X1)
    	return y_pred_proba
