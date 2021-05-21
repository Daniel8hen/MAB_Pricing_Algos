import numpy as np
from types import MethodType

class decision_function_from_proba:
    def __init__(self, model):
        self.model = model
    def __call__(self, f):
        def new_f(this, X):
            proba=f(this.predict_proba(X))
            return -np.log(proba)
        self.model.decision_function=MethodType(new_f, self.model)
        return self.model
