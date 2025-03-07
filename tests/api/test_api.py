import unittest

from fastapi.testclient import TestClient
from challenge import app


class TestBatchPipeline(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_should_get_predict(self):
        data = {
            "flights": [{"OPERA": "Aerolineas Argentinas", "TIPOVUELO": "N", "MES": 3}]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0])) # change this line to the model of chosing
        with self.client:
            response = self.client.post("/predict", json=data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predict": [0]})

    def test_should_get_predict_for_multiple_examples(self):
        data = {
            "flights": [
                {"OPERA": "Aerolineas Argentinas", "TIPOVUELO": "N", "MES": 1},
                {"OPERA": "Grupo LATAM", "TIPOVUELO": "N", "MES": 2},
                {"OPERA": "Sky Airline", "TIPOVUELO": "I", "MES": 3},
                {"OPERA": "Copa Air", "TIPOVUELO": "I", "MES": 4},
            ]
        }

        with self.client:
            response = self.client.post("/predict", json=data)

        self.assertEqual(response.status_code, 200)

    def test_should_failed_empty_data(self):
        data = {"flights": []}
        with self.client:
            response = self.client.post("/predict", json=data)

        self.assertEqual(response.status_code, 400)

    def test_should_failed_unkown_column_1(self):
        data = {
            "flights": [{"OPERA": "Aerolineas Argentinas", "TIPOVUELO": "N", "MES": 13}]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))# change this line to the model of chosing
        with self.client:
            response = self.client.post("/predict", json=data)

        self.assertEqual(response.status_code, 400)

    def test_should_failed_unkown_column_2(self):
        data = {
            "flights": [{"OPERA": "Aerolineas Argentinas", "TIPOVUELO": "O", "MES": 13}]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))# change this line to the model of chosing
        with self.client:
            response = self.client.post("/predict", json=data)

        self.assertEqual(response.status_code, 400)

    def test_should_failed_unkown_only_column_2(self):
        data = {
            "flights": [{"OPERA": "Aerolineas Argentinas", "TIPOVUELO": "O", "MES": 3}]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))# change this line to the model of chosing
        with self.client:
            response = self.client.post("/predict", json=data)

        self.assertEqual(response.status_code, 400)

    def test_should_failed_unkown_column_3(self):
        data = {"flights": [{"OPERA": "Argentinas", "TIPOVUELO": "O", "MES": 13}]}
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))
        with self.client:
            response = self.client.post("/predict", json=data)

        self.assertEqual(response.status_code, 400)

    def test_should_failed_unkown_only_column_3(self):
        data = {"flights": [{"OPERA": "Argentinas", "TIPOVUELO": "I", "MES": 3}]}
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))
        with self.client:
            response = self.client.post("/predict", json=data)

        self.assertEqual(response.status_code, 400)
