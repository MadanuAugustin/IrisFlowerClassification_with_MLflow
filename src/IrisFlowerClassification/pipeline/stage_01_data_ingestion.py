
import sys
from src.IrisFlowerClassification.config.configuration import ConfigurationManager
from src.IrisFlowerClassification.components.data_ingestion import DataIngestion
from logger_file.logging import logger
from Exception_file.exception import CustomException

STAGE_NAME = 'Data_Ingestion_Stage'


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()