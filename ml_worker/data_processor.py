import json
import numpy as np
import pandas as pd
from typing import Iterable
from flask_restx import Model
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


models_list = Model(
    'models list',
    {
        'Logistic Regression': {
            'model': LogisticRegression,
            'hyperparams': {'C': 'float (>0)'},
            'default_hyperparams': {'C': 1},
        },
        'Random Forest': {
            'model': RandomForestClassifier,
            'hyperparams': {'n_estimators': 'int (>0)', 'max_depth': 'int (>0)'},
            'default_hyperparams': {'n_estimators': 100, 'max_depth': None}
        },
    },
)


class DataProcessor:

    @staticmethod
    def get_model_report(x, y, model):
        cv_result = cross_validate(
            model, x, y, cv=5, scoring=('precision', 'recall', 'roc_auc', 'f1')
        )
        cv_metrics = []
        for score_name, scores in cv_result.items():
            if 'time' in score_name:
                continue
            cv_metrics.append(np.mean(scores).round(3))
        report = pd.DataFrame(
            cv_metrics, index=['precision', 'recall', 'roc_auc', 'f1_score'], columns=['score']
        )
        return report.to_json()

    def preprocess_train_data(self, train_data: str):
        train_dataframe = pd.DataFrame(json.loads(train_data))
        x, y = train_dataframe.drop(columns='target'), train_dataframe['target']
        column_names = x.columns
        categorical_features = column_names[self._find_categorical_columns(column_names)]
        if len(categorical_features) > 0:
            x = pd.get_dummies(x, columns=categorical_features)
        return x, y

    def preprocess_prediction_data(self, prediction_data: str):
        x = pd.DataFrame(json.loads(prediction_data))
        column_names = x.columns
        categorical_features = column_names[self._find_categorical_columns(column_names)]
        if len(categorical_features) > 0:
            x = pd.get_dummies(x, columns=categorical_features)
        return x

    @staticmethod
    def _find_categorical_columns(column_names: Iterable):
        categorical_index = []
        for name in column_names:
            if 'categorical' in name:
                categorical_index.append(True)
            else:
                categorical_index.append(False)
        return categorical_index
