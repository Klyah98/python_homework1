import os

from celery import Celery
from loguru import logger

CELERY_BROKER = os.environ["CELERY_BROKER"]
CELERY_BACKEND = os.environ["CELERY_BACKEND"]

celery = Celery("tasks", broker=CELERY_BROKER, backend=CELERY_BACKEND)


class MLModelsDAO:
    def __init__(self):
        self.ml_models = {}

    def get(self, task_id):
        logger.debug(f"Status request for task_id {task_id}")
        result = celery.AsyncResult(task_id)
        if result.status == "PENDING":
            return "PENDING"
        else:
            return result.result

    def get_models_info(self):
        task_id = celery.send_task(
            "models_info",
            args=[],
        )
        logger.debug(f"Send task about models, {task_id=}")
        return str(task_id)

    def create(self, model_name, model_type, hyperparams, train_data):
        task_id = celery.send_task(
            "train_model",
            args=[model_name, model_type, hyperparams, train_data],
        )
        logger.info(f"Send task to train new model, {task_id=}")
        return str(task_id)

    def update(self, model_name, new_data):
        task_id = celery.send_task(
            "retrain_model",
            args=[model_name, new_data],
        )
        logger.info(f"Send task to update existing model, {task_id=}")
        return str(task_id)

    def delete(self, model_name):
        task_id = celery.send_task(
            "delete_model",
            args=[model_name],
        )
        logger.info(f"Send task to delete existing model, {task_id=}")
        return str(task_id)

    def predict(self, model_name, predict_data, return_proba):
        task_id = celery.send_task(
            "get_predictions",
            args=[model_name, predict_data, return_proba],
        )
        logger.info(
            f"Send task to make predictions on data using {model_name}, {task_id=}"
        )
        return str(task_id)
