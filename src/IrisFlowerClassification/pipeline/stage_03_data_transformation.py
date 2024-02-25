
from pathlib import Path
from src.IrisFlowerClassification.config.configuration import ConfigurationManager
from src.IrisFlowerClassification.components.data_transformation import DataTransformation
from logger_file.logging import logger
from Exception_file.exception import CustomException
import sys


STAGE_NAME = 'Data_Transformation_Stage'

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        try:
            with open(Path('artifacts//data_validation//status.txt'), 'r') as f:
                status = f.read().split(" ")[-1]

            
            if status == 'True':
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config = data_transformation_config)
                data_transformation.train_test_splitting()

            else:
                print('Your data schema is not valid...!')

        except Exception as e:
            raise CustomException(e, sys)
        