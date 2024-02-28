

from src.IrisFlowerClassification.entity.config_entity import ModelTrainerConfig
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
import os



class ModelTrainer:
    def __init__(self, config : ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis = 1)
        test_x = test_data.drop([self.config.target_column], axis = 1)

        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        classifier = DecisionTreeClassifier(criterion=self.config.criterion, max_depth=self.config.max_depth, max_features=self.config.max_features)

        classifier.fit(train_x, train_y)

        joblib.dump(classifier, os.path.join(self.config.root_dir, self.config.model_name))