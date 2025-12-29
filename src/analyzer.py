import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class EpilepsyDataAnalyzer:
    def __init__(self, input_data_path):
        self.data = pd.read_csv(input_data_path)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)

    def preprocess_data(self):
        # Handle missing values
        self.data.fillna(method='ffill', inplace=True)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(self.data.drop(columns=['PatientID']))
        return scaled_data

    def apply_pca(self, scaled_data):
        # Apply PCA
        pca_result = self.pca.fit_transform(scaled_data)
        return pca_result

    def analyze_data(self):
        scaled_data = self.preprocess_data()
        pca_result = self.apply_pca(scaled_data)
        self.data['PCA1'] = pca_result[:, 0]
        self.data['PCA2'] = pca_result[:, 1]
        return self.data

    def save_results(self, output_path):
        analyzed_data = self.analyze_data()
        analyzed_data.to_json(output_path, orient='records')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze patient data for EpilepsyCellCleaner")
    parser.add_argument('--input-data', type=str, required=True, help='Path to the input CSV data file')
    parser.add_argument('--output-results', type=str, required=True, help='Path to save the output JSON results file')
    args = parser.parse_args()

    analyzer = EpilepsyDataAnalyzer(args.input_data)
    analyzer.save_results(args.output_results)