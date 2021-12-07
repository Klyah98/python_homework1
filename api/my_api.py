from flask import Flask, request
from flask_restx import Api, Resource
from models_dao import MLModelsDAO
from dto import models_list, TrainInput, PredictInput


app = Flask(__name__)
api = Api(app)


models_dao = MLModelsDAO()
# @api.route('/test_api/ml_models')
# class MLModelsInfo(Resource):
#
#     @staticmethod
#     def _create_description():
#         report = 'here is a list of available models:'
#         for model in models_list.keys():
#             hyperparams = models_list[model]['hyperparams']
#             s = f' - {model}, allowed hyperparameters: {hyperparams};'
#             report = report + '\n' + s
#         return report
#
#     def get(self):
#         models_report = self._create_description()
#         return models_report


@api.route('/test_api/ml_models/task_result')
class MLModelsTrain(Resource):

    def get(self):
        """get information about task for models"""
        task_id = request.args.get('task_id')
        return models_dao.get(task_id)


@api.route('/test_api/ml_models/train')
class MLModelsTrain(Resource):

    def get(self):
        """get information about trained models"""
        return models_dao.get_models_info()

    def post(self):
        """train new model with input data"""
        input_data = TrainInput.parse_obj(api.payload)
        report = models_dao.create(
            input_data.model_name,
            input_data.model_type,
            input_data.hyperparams,
            input_data.train_data,
        )
        return report

    def put(self):
        """retrain existing model with new data"""
        model_name = request.form['model_name']
        new_data = request.form['new_data']
        report = models_dao.update(model_name, new_data)
        return report


@api.route('/test_api/ml_models/delete/<string:model_name>')
class MLModelsDelete(Resource):

    def delete(self, model_name):
        """delete existing model from models list"""
        result = models_dao.delete(model_name)
        return result


@api.route('/test_api/ml_models/predict')
class MLModelsPredict(Resource):

    def post(self):
        """get prediction via trained model"""
        input_data = PredictInput.parse_obj(api.payload)
        answers = models_dao.predict(
            input_data.model_name,
            input_data.predict_data,
            input_data.return_proba,
        )
        return answers


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
