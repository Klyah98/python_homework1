import pytest
from api.my_api import app


TRAIN_URL = 'ml_models/train'


def test_api():
    with app.test_client() as client:
        response = client.get(TRAIN_URL)
        assert response is None
