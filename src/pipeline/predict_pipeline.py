import pandas as pd
import sys
from src.exception import CustomException
from src.utlis import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)

            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)

            return prediction
            
        except Exception as e:
            raise CustomException(e,sys)
        



class CustomData:
    """
    this class will be responsible in mapping all the inputs that we are giving in the HTML to the backend with this particular values

    """
    def __init__(self,
        gender:str,
        race_ethnicity:str,
        parental_level_of_education,
        lunch:str,
        test_preparation_course:str,
        reading_score:int,
        writing_score:int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    def get_data_as_dataframe(self):
        """
        this function will return all the inputs in the form of the dataframe becuase the we train our models in a form of the dataframe 
        """
        try:
            custom_data_input_dict = {
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e.sys)




