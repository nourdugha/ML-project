"""
Here all the common methods that might be used in any file in the project

"""
import os
import sys
# dill lib for Serialization (also known as "pickling"), and for deserialization (or "unpickling")
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train,y_train)

            y_train_prediction = model.predict(X_train)

            y_test_prediction = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_prediction)

            test_model_score = r2_score(y_test,y_test_prediction)

            report[list(models.keys())[i]] = test_model_score

            return report


    except Exception as e:
        raise CustomException(e,sys)

