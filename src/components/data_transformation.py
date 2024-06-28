"""
 main purpose :
 (Feature Engineering, Data cleaning, Handling categorical features and numrical one).

"""
import sys
import os
import numpy as np
import pandas as pd
from src.utlis import save_object
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging

# providing all the input things that is required for this data transformation


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ["reading_score", "writing_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # numerical pipeline (needs to run on the train and test datasets)
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            # categorical pipeline (needs to run on the train and test datasets)
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical columns standard scaling")
            logging.info("categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "math_score"

            # X_train
            input_features_train_df = train_df.drop(
                columns=[target_column_name], axis=1)
            # y_train
            traget_features_train_df = train_df[target_column_name]

            # X_test
            input_features_test_df = test_df.drop(
                columns=[target_column_name], axis=1)
            # y_test
            traget_features_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and test dataframe")

            input_features_train_arr = preprocessor_obj.fit_transform(
                input_features_train_df)
            input_features_test_arr = preprocessor_obj.transform(
                input_features_test_df)

            train_arr = np.c_[input_features_train_arr,
                              np.array(traget_features_train_df)]
            test_arr = np.c_[input_features_test_arr,
                             np.array(traget_features_test_df)]

            logging.info(f"Saved preprocessing object")

            # for saving the pickle file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessor_obj)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
