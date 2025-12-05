import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

class CrisisEngine:
    def __init__(self):
        
        # Load the model
        cache_path = os.path.join(os.getcwd(), "model_cache")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_path)
        
        
        # Load the crisis data efficiently with pandas
        csv_path = 'data/train.csv'
        sample_size = 200
        try:
            # Optimization: Load only necessary columns
            # This reduces memory footprint significantly if the CSV has many columns
            req_cols = ['text', 'location']
            
            # Optimization: Calculate file size to sample without loading everything
            # (If file is small <10MB, normal read_csv is fine. This is for larger files)
            total_rows = sum(1 for _ in open(csv_path, 'r', encoding='utf-8', errors='replace')) - 1
            
            if total_rows > sample_size:
                # Generate random indices to keep
                skip = sorted(np.random.choice(np.arange(1, total_rows + 1), (total_rows - sample_size), replace=False))
                self.df = pd.read_csv(
                    csv_path, 
                    usecols=req_cols, 
                    skiprows=skip,
                    encoding='utf-8', 
                    on_bad_lines='skip'
                )
            else:
                self.df = pd.read_csv(csv_path, usecols=req_cols, encoding='utf-8', on_bad_lines='skip')
            
            # Ensure we handle missing locations gracefully
            self.df['location'] = self.df['location'].fillna('Unknown')
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            self.df = pd.DataFrame(columns=['text', 'location'])

        
        if not self.df.empty:
            # Pre_calculate vectors
            self.vectors = self.model.encode(self.df['text'].tolist(), show_progress_bar=True)
            
            # Perform Clustering
            # We tell it to find 5 "Themes" in the data
            self.kmeans = KMeans(n_clusters=5, random_state=42)
            self.clusters = self.kmeans.fit_predict(self.vectors)
            
            # Perform PCA (Dimensionality Reduction for Visualization)
            # Squash 384 dims -> 2 dims (x, y)
            self.pca = PCA(n_components=2)
            self.coords = self.pca.fit_transform(self.vectors)
        else:
            self.vectors = []
            self.clusters = []
            self.coords = []
        # Process Data (Vectorization -> Clustering -> PCA)
        if not self.df.empty:
            self.vectors = self.model.encode(self.df['text'].tolist(), show_progress_bar=False)
            
            self.kmeans = KMeans(n_clusters=5, random_state=42)
            self.clusters = self.kmeans.fit_predict(self.vectors)
            
            self.pca = PCA(n_components=2)
            self.coords = self.pca.fit_transform(self.vectors)
        else:
            self.vectors, self.clusters, self.coords = [], [], []
        
    def get_dashboard_data(self):
        """
        Returns data ready for the Frontend Map & Charts
        """
        results = []
        if self.df.empty:
            return results

        # Pandas iterrows is slow for huge data, but fine for 200 rows
        for i, row in self.df.iterrows():
            results.append({
                "id": int(i),
                "text": row['text'],
                "location": row['location'],
                "cluster_id": int(self.clusters[i]),
                "pca_x": float(self.coords[i][0]), 
                "pca_y": float(self.coords[i][1]),
            })
        return results