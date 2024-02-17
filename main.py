from logger_file.logging import logger
from src.IrisFlowerClassification.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from Exception_file.exception import CustomException
import sys






STAGE_NAME = 'Data_Ingestion_Stage'


try:
    logger.info(f'-----------{STAGE_NAME} started----------------')
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f'---------------{STAGE_NAME} completed--------------')

except Exception as e:
    raise CustomException(e, sys)