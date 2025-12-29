import unittest
from predictor import Predictor

class TestPredictor(unittest.TestCase):

    def setUp(self):
        self.predictor = Predictor()

    def test_cellular_aging_prediction(self):
        result = self.predictor.predict_ageing_patterns()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)

    def test_model_type_validation(self):
        with self.assertRaises(ValueError):
            self.predictor.set_model_type('invalid_model')

    def test_valid_model_type(self):
        self.predictor.set_model_type('cellular_aging')
        self.assertEqual(self.predictor.model_type, 'cellular_aging')

if __name__ == '__main__':
    unittest.main()