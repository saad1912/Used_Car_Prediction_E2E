import pandas as pd
import os
from mlProject import logger
from catboost import CatBoostRegressor
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        categorical_columns = train_x.select_dtypes(include=['object']).columns
        for col in categorical_columns:
                # Convert categorical column to category codes
            train_x[col] = train_x[col].astype('category').cat.codes
            test_x[col] = test_x[col].astype('category').cat.codes
        
        lr = CatBoostRegressor(max_depth=self.config.max_depth, learning_rate=self.config.learning_rate, random_state=42,
                               cat_features=categorical_columns.tolist())
        lr.fit(train_x, train_y,cat_features=categorical_columns.tolist())

        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))