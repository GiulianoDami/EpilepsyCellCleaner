import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        try:
            data = pd.read_csv(self.file_path)
            return data
        except FileNotFoundError:
            print(f"Error: The file {self.file_path} was not found.")
            return None
        except pd.errors.EmptyDataError:
            print(f"Error: The file {self.file_path} is empty.")
            return None
        except pd.errors.ParserError:
            print(f"Error: The file {self.file_path} could not be parsed.")
            return None

    def validate_data(self, data):
        if data is None:
            return False
        required_columns = ['patient_id', 'age', 'biomarker1', 'biomarker2', 'biomarker3']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"Error: The following columns are missing from the data: {missing_columns}")
            return False
        return True