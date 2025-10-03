import codecs
import os
# import sys
import unittest
from fastapi.testclient import TestClient
from main import app
# sys.path.append(os.path.abspath('./app'))
client = TestClient(app)

class TestApp(unittest.TestCase):

    def test_read_root(self):
        # Teste la route "/" qui renvoie le contenu de presentation.html
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        presentation_path = os.path.join(base_dir, 'presentation.html')
        with codecs.open(presentation_path, 'r', 'utf-8') as f:
            expected_html = f.read()
        self.assertEqual(response.content.decode('utf-8'), expected_html)

    def test_validate_ticker(self):
        # Teste la validation d'un ticker (ici "AAPL")
        response = client.get("/validate_ticker/AAPL")
        self.assertEqual(response.status_code, 200)
        json_resp = response.json()
        self.assertIn("ticker", json_resp)
        self.assertIn("valid", json_resp)
        # On s'attend à ce qu'AAPL soit un ticker valide
        self.assertTrue(json_resp["valid"])

    def test_simulate(self):
        # Teste l'endpoint "/simulate"
        payload = {
            "ticker": "AAPL",
            "quantity": 100,
            "riskFreeRate": 0.05,
            "date": "01/01/2023",
            "maturityDate": "06/01/2023",
            "strike": 150,
            "rebalancing_freq": 12,
            "current_underlying_weight": 0,
            "current_cash": 0
        }
        response = client.post("/simulate", json=payload)
        self.assertEqual(response.status_code, 200)
        json_resp = response.json()
        # La réponse doit contenir la clé "prediction"
        self.assertIn("prediction", json_resp)

    def test_compare_strategies(self):
        # Teste l'endpoint "/compare_strategies"
        payload = {
            "ticker": "AAPL",
            "start_date": "01/01/2023",
            "maturity_date": "01/01/2024",
            "quantity": 150,
            "risk_free_rate": 0.05,
            "strike": 100,
            "rebalance_freq": 12,
            "initial_weights": [0, 0]
        }
        response = client.post("/compare_strategies", json=payload)
        self.assertEqual(response.status_code, 200)
        json_resp = response.json()
        self.assertIn("results", json_resp)
        self.assertIn("alert", json_resp)
        # On s'attend à ce que l'alerte soit None (aucune erreur)
        self.assertIsNone(json_resp["alert"])

    def test_metrics(self):
        # Teste l'endpoint "/metrics" de Prometheus
        response = client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        # Vérifier que le contenu renvoyé contient le compteur "total_requests"
        self.assertIn("total_requests", response.text)

if __name__ == '__main__':
    unittest.main()
