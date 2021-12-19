from dto import PredictInput, TrainInput
from flask import Flask, request
from flask_restx import Api, Resource
from loguru import logger
from models_dao import MLModelsDAO
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)
api = Api(app)
metrics = PrometheusMetrics(app)

logger.add(
    "/log_files/api_logging.log",
    format="{time} {level} {message}",
    level="DEBUG",
    rotation="5 MB",
    compression="zip",
    serialize=True,
)

models_dao = MLModelsDAO()


@api.route("/test_api/ml_models/task_result")
class MLModelsTrain(Resource):
    @logger.catch
    def get(self):
        """get information about task for models"""
        task_id = request.args.get("task_id")
        return models_dao.get(task_id)


@api.route("/test_api/ml_models/train")
class MLModelsTrain(Resource):
    @logger.catch
    def get(self):
        """get information about trained models"""
        return models_dao.get_models_info()

    @logger.catch
    @metrics.counter(
        "cnt_trains",
        "some_desc",
        labels={"status": lambda response: response.status_code},
    )
    def post(self):
        """train new model with input data"""
        input_data = TrainInput.parse_obj(api.payload)
        logger.debug(
            f"Request fot for training new {input_data.model_name}: {input_data}"
        )
        report = models_dao.create(
            input_data.model_name,
            input_data.model_type,
            input_data.hyperparams,
            input_data.train_data,
        )
        return {"report": report}

    @logger.catch
    @metrics.counter(
        "cnt_retrains",
        "some_desc",
        labels={"status": lambda response: response.status_code},
    )
    def put(self):
        """retrain existing model with new data"""
        model_name = request.form["model_name"]
        new_data = request.form["new_data"]
        logger.debug(f"Request for retraining {model_name} wit new data")
        report = models_dao.update(model_name, new_data)
        return {"report": report}


@api.route("/test_api/ml_models/delete/<string:model_name>")
class MLModelsDelete(Resource):
    @logger.catch
    @metrics.counter(
        "cnt_deletes",
        "some_desc",
        labels={"status": lambda response: response.status_code},
    )
    def delete(self, model_name):
        """delete existing model from models list"""
        logger.info(f"Request for delete {model_name}")
        result = models_dao.delete(model_name)
        return {"result": result}


@api.route("/test_api/ml_models/predict")
class MLModelsPredict(Resource):
    @logger.catch
    @metrics.counter(
        "cnt_predicts",
        "some_desc",
        labels={"status": lambda response: response.status_code},
    )
    def post(self):
        """get prediction via trained model"""
        input_data = PredictInput.parse_obj(api.payload)
        logger.info(f"Request predicting, input data: {input_data}")
        answers = models_dao.predict(
            input_data.model_name,
            input_data.predict_data,
            input_data.return_proba,
        )
        return {"answers": answers}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
