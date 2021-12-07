import os
from celery import Celery

CELERY_BROKER = os.environ['CELERY_BROKER']
CELERY_BACKEND = os.environ['CELERY_BACKEND']

celery = Celery('tasks', broker=CELERY_BROKER, backend=CELERY_BACKEND)


class MLModelsDAO:
    def __init__(self):
        self.ml_models = {}

    def get(self, task_id):
        result = celery.AsyncResult(task_id)
        if result.status == 'PENDING':
            return 'PENDING'
        else:
            return result.result

    def get_models_info(self):
        task_id = celery.send_task(
            'models_info',
            args=[],
        )
        return str(task_id)

    def create(self, model_name, model_type, hyperparams, train_data):
        task_id = celery.send_task(
            'train_model',
            args=[model_name, model_type, hyperparams, train_data],
        )
        return str(task_id)

    def update(self, model_name, new_data):
        task_id = celery.send_task(
            'retrain_model',
            args=[model_name, new_data],
        )
        return str(task_id)

    def delete(self, model_name):
        task_id = celery.send_task(
            'delete_model',
            args=[model_name],
        )
        return str(task_id)

    def predict(self, model_name, predict_data, return_proba):
        task_id = celery.send_task(
            'get_predictions',
            args=[model_name, predict_data, return_proba],
        )
        return str(task_id)
