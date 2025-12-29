import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class CellularAgingModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, data_path):
        data = pd.read_csv(data_path)
        X = data.drop('senescence', axis=1)
        y = data['senescence']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        print(f"Model Accuracy: {accuracy_score(y_test, predictions)}")

    def predict(self, new_data):
        return self.model.predict(new_data)

    def save_model(self, model_path):
        import joblib
        joblib.dump(self.model, model_path)

    def load_model(self, model_path):
        import joblib
        self.model = joblib.load(model_path)