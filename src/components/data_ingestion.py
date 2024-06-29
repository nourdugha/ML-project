"""
main purpose : 
1- read our dataset from some kind of source.
2- did train test split.
3- save all kinds of data (raw, train, test) in the artifacts folder.

"""

from pathlib import Path
import sys
import os
import pandas as pd
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


"""
The @dataclass decorator is used to automatically generate special methods for the class,
 such as __init__, __repr__, and __eq__, based on the class attributes.
 This simplifies the class definition by removing the need to write these methods manually.

"""

# location of the dataset
location = r"src\notebook\data\stud.csv"


@dataclass
class DataIngestionConfig:
    """
    A dataclass that defines paths where the raw, training, and testing datasets will be saved.
    """
    train_data_path: str = os.path.join(
        'artifacts', 'train.csv')  # 'artifacts\train.csv'
    test_data_path: str = os.path.join(
        'artifacts', 'test.csv')  # 'artifacts\test.csv'
    raw_data_path: str = os.path.join(
        'artifacts', 'raw.csv')  # 'artifacts\raw.csv'


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(location)
            logging.info('read the dataset as dataframe')

            # Creates the directory for storing the datasets if it does not already exist.
            os.makedirs(os.path.dirname(
                self.ingestion_config.train_data_path), exist_ok=True)

            # Saves the raw dataset to the path specified in raw_data_path without row indices and with headers.
            df.to_csv(self.ingestion_config.raw_data_path,
                      index=False, header=True)

            logging.info("Train test split initiate")
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42)

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,
                            index=False, header=True)

            logging.info("Ingestion of the Dataset is completed")

            # these two things will need them to the data transformation
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_= data_transformation.initiate_data_transformation(
        train_path=train_data, test_path=test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr=train_arr,test_arr=test_arr))
