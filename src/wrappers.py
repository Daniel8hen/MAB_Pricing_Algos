import numpy as np
from types import MethodType
from sklearn.ensemble import BaggingClassifier
from copy import deepcopy as clone

class decision_function_from_proba:
    def __init__(self, model):
        self.model = model
    def __call__(self, f):
        def new_f(this, X):
            proba=f(this.predict_proba(X))
            return -np.log(proba)
        self.model.decision_function=MethodType(new_f, self.model)
        return self.model

class TemporalBagging(BaggingClassifier):
    def __init__(self, base_classifier, n_estimators, **kwargs)
        super().__init__(base_classifier, n_estimators, **kwargs)
        self.current_index = 0
    def partial_fit(self, X, y):
        self.estimators_[self.current_index].fit(X,y)
        self.current_index = (self.current_index + 1) % len(self.estimators_)
