
from src.IrisFlowerClassification.config.configuration import ConfigurationManager
from src.IrisFlowerClassification.components.model_evaluation import ModelEvaluation
from logger_file.logging import logger








STAGE_NAME = 'ModelEvaluationStage'


class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.log_into_mlflow()


        