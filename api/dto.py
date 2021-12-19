from flask_restx import Model
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

models_list = Model(
    "models list",
    {
        "Logistic Regression": {
            "model": LogisticRegression,
            "hyperparams": {"C": "float (>0)"},
            "default_hyperparams": {"C": 1},
        },
        "Random Forest": {
            "model": RandomForestClassifier,
            "hyperparams": {"n_estimators": "int (>0)", "max_depth": "int (>0)"},
            "default_hyperparams": {"n_estimators": 100, "max_depth": None},
        },
    },
)


class PredictInput(BaseModel):
    model_name: str
    return_proba: bool = False
    predict_data: str


class TrainInput(BaseModel):
    model_name: str
    model_type: str = "Logistic Regression"
    hyperparams: dict = {}
    train_data: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.hyperparams == {}:
            self.hyperparams = models_list[self.model_type]["default_hyperparams"]
