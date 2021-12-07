import os
import pickle
from celery import Celery
from data_processor import DataProcessor, models_list
from mongo_client import MongoDBClient

MONGO_DB_HOST = os.environ['MONGO_DB_HOST']
MONGO_DB_PORT = os.environ['MONGO_DB_PORT']
CELERY_BROKER = os.environ['CELERY_BROKER']
CELERY_BACKEND = os.environ['CELERY_BACKEND']


celery = Celery('tasks', broker=CELERY_BROKER, backend=CELERY_BACKEND)
mongo_db = MongoDBClient(
    host=MONGO_DB_HOST,
    port=int(MONGO_DB_PORT),
)
data_processor = DataProcessor()


@celery.task(name='train_model')
def train_model(model_name, model_type, hyperparams, train_data):
    x, y = data_processor.preprocess_train_data(train_data)
    model = models_list[model_type]['model'](**hyperparams)
    report = data_processor.get_model_report(x, y, model)
    model.fit(x, y)
    mongo_db.create(model_name, model)
    return report


@celery.task(name='retrain_model')
def retrain_model(model_name, new_data):
    x, y = data_processor.preprocess_train_data(new_data)
    model = pickle.loads(mongo_db.read(model_name)['model'])
    report = data_processor.get_model_report(x, y, model)
    model.fit(x, y)
    mongo_db.update(model_name, model)
    return report


@celery.task(name='delete_model')
def delete_model(model_name):
    mongo_db.delete(model_name)
    return 'Successfully deleted'


@celery.task(name='get_predictions')
def get_predictions(model_name, predict_data, return_proba):
    predict_dataframe = data_processor.preprocess_prediction_data(predict_data)
    model = pickle.loads(mongo_db.read(model_name)['model'])
    if bool(return_proba):
        return model.predict_proba(predict_dataframe)[:, 1].tolist()
    else:
        return model.predict(predict_dataframe).tolist()


@celery.task(name='models_info')
def models_info():
    return mongo_db.get_documents_info()
