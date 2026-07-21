import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class PredictionModel:
    def __init__(self,
                 model = LogisticRegression(),
                 scaler = StandardScaler(),
                 ):
        self.model = model
        self.scaler = scaler if scaler else None

        self.trained = False
        self.label_name = None
        self.feature_names = None
        return

    def train(self, X_train, y_train):
        self.label_name = y_train.name
        self.feature_names = X_train.columns

        if self.scaler is not None:
            X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)
        self.trained = True
        return

    def predict(self, X_test):
        if not self.trained:
            raise Exception('Prediction model not trained')
        index = X_test.index
        if self.scaler is not None:
            X_test = self.scaler.transform(X_test)

        y_pred = self.model.predict(X_test) # in numpy array
        return pd.Series(y_pred,
                         name=self.label_name,
                         index=index,)

    def predict_proba(self, X_test):
        if not self.trained:
            raise Exception('Prediction model not trained')
        index = X_test.index
        if self.scaler is not None:
            X_test = self.scaler.transform(X_test)

        y_pred = self.model.predict_proba(X_test) # in numpy array
        y_pred = y_pred[:, 1] # prob of label 1
        return pd.Series(y_pred,
                         name=self.label_name,
                         index=index,)

    def get_model_coeff(self):
        if not self.trained:
            raise Exception('Prediction model not trained')
        # coef_ is (1, n_features) for classifiers, (n_features,) for
        # regressors; ravel flattens both to 1-D
        coefs = pd.Series(np.ravel(self.model.coef_), index=self.feature_names)
        return coefs