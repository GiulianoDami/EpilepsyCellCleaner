import unittest
from predictor import CellularAgingModel

class TestCellularAgingModel(unittest.TestCase):

    def setUp(self):
        self.model = CellularAgingModel()

    def test_predict_ageing_patterns(self):
        test_data = {
            'biomarker1': [0.5],
            'biomarker2': [0.3],
            'biomarker3': [0.8]
        }
        result = self.model.predict_ageing_patterns(test_data)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)

    def test_invalid_input(self):
        invalid_data = "not a dictionary"
        with self.assertRaises(TypeError):
            self.model.predict_ageing_patterns(invalid_data)

    def test_missing_biomarker(self):
        incomplete_data = {
            'biomarker1': [0.5],
            'biomarker2': [0.3]
        }
        with self.assertRaises(KeyError):
            self.model.predict_ageing_patterns(incomplete_data)

if __name__ == '__main__':
    unittest.main()