from flask import Flask, request
from flask_restx import Api, Resource
from dto import models_list, TrainInput, PredictInput
from data_processor import DataProcessor


app = Flask(__name__)
api = Api(app)


class MLModelsDAO:
    def __init__(self):
        self.ml_models = {}
        self.data_processor = DataProcessor()

    def get(self):
        report = {}
        for model_name, model in self.ml_models.items():
            report[model_name] = str(model)
        return report

    def create(self, model_name, model_type, hyperparams, train_data):
        x, y = self.data_processor.preprocess_train_data(train_data)
        model = models_list[model_type]['model'](**hyperparams)
        report = self.data_processor.get_model_report(x, y, model)
        model.fit(x, y)
        self.ml_models[model_name] = model
        return report

    def update(self, model_name, new_data):
        x, y = self.data_processor.preprocess_train_data(new_data)
        model = self.ml_models[model_name]
        report = self.data_processor.get_model_report(x, y, model)
        model.fit(x, y)
        return report

    def delete(self, model_name):
        del self.ml_models[model_name]

    def predict(self, model_name, predict_data, return_proba):
        predict_dataframe = self.data_processor.preprocess_prediction_data(predict_data)
        print(self.ml_models)
        model = self.ml_models[model_name]
        if return_proba:
            return model.predict_proba(predict_dataframe)[:, 1].tolist()
        else:
            return model.predict(predict_dataframe).tolist()


models_dao = MLModelsDAO()


@api.route('/test_api/ml_models')
class MLModelsInfo(Resource):

    @staticmethod
    def _create_description():
        report = 'here is a list of available models:'
        for model in models_list.keys():
            hyperparams = models_list[model]['hyperparams']
            s = f' - {model}, allowed hyperparameters: {hyperparams};'
            report = report + '\n' + s
        return report

    def get(self):
        models_report = self._create_description()
        return models_report


@api.route('/test_api/ml_models/train')
class MLModelsTrain(Resource):

    def get(self):
        """get information about trained models"""
        return models_dao.get()

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
        models_dao.delete(model_name)


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
    app.run(debug=True)
