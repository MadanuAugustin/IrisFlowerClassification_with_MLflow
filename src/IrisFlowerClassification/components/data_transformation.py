from src.IrisFlowerClassification.entity.config_entity import DataTransformationConfig
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from logger_file.logging import logger
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

class DataTransformation:
    def __init__(self, config : DataTransformationConfig):
        self.config = config


    def train_test_splitting(self):
        data = pd.read_csv(self.config.data_path)

        logger.info('splitting the data into training and testing...!')

        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, 'train_raw.csv'), index = False, header = True)
        test.to_csv(os.path.join(self.config.root_dir, 'test_raw.csv'), index = False, header = True)

        train.drop('Id', axis= 1, inplace = True)

        train_X = train.drop(columns='Species', axis = 1)

        train_Y = train[['Species']]

        std = StandardScaler()

        train_X = std.fit_transform(train_X)

        train_X = pd.DataFrame(train_X)

        le = LabelEncoder()

        train_Y = le.fit_transform(train_Y)

        train_Y = pd.DataFrame(train_Y)

        train_Y.rename(columns={0 : 'Species'}, inplace=True)

        transformed_train = pd.concat([train_X, train_Y], axis = 1)

        transformed_train.to_csv(os.path.join(self.config.root_dir, 'transformed_train_data.csv'), index = False, header = True)

        test.drop('Id', axis = 1, inplace = True)

        test_X = test.drop(columns = 'Species', axis = 1)

        test_Y = test[['Species']]

        test_X = std.transform(test_X)

        test_X = pd.DataFrame(test_X)

        test_Y = le.transform(test_Y)

        test_Y = pd.DataFrame(test_Y)

        test_Y.rename(columns={0 : 'Species'}, inplace= True)

        transformed_test = pd.concat([test_X, test_Y], axis = 1)

        transformed_test.to_csv(os.path.join(self.config.root_dir, 'transformed_test_data.csv'), index = False, header = True)

        logger.info('completed splitting the data into training and testing...!')
        logger.info(transformed_train.shape)
        logger.info(transformed_test.shape)

        print(transformed_train.shape)
        print(transformed_test.shape)

